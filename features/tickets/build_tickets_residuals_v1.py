"""Build features/tickets/tickets_residuals_v1.ipynb: do survival residuals from a model without ticket information
cluster by ticket number?

Logistic regression on every non-ticket feature, held-out predictions from 10 x 10-fold CV averaged per passenger,
then residuals tested by ticket number (prefix ignored), consecutive-number runs, number ranges, and residual
correlation against number distance. Reads data/complete_data.csv. Code cells hidden.
"""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
AREA = ROOT / "features" / "tickets"
HIDE = {"jupyter": {"source_hidden": True}}
FINDINGS = {}
if (Path(__file__).resolve().parent / "tickets_residuals_v1_findings.py").exists():
    exec((Path(__file__).resolve().parent / "tickets_residuals_v1_findings.py").read_text())

intro = """# Ticket number and model residuals

Does the ticket number explain survival that the other features cannot?

**1. Model without the ticket.** Logistic regression on every feature except the ticket: class, sex, title, age (median-filled, with a flag), SibSp, Parch, fare (log), port and deck. Deck comes from the passenger's own cabin only, not from ticket-mates, so no ticket information enters.

**2. Held-out residuals.** 10-fold cross-validation, repeated 10 times with different shuffles. Each train passenger gets 10 predictions, each from a model that never saw them, and these are averaged. The residual is survived (1 or 0) minus that prediction: positive means the passenger did better than the model expected.

**3. Ticket groupings, prefix ignored.** Only the number is used, so `PC 17474` and `17474` count as the same number, and `STON/O2. 3101294` and `SOTON/O.Q. 3101295` as consecutive:

- **Exact ticket**: passengers holding the same ticket (prefix included), for comparison.
- **Consecutive run**: tickets whose numbers follow on with gaps of at most 1 (or at most 3) are chained into one run.
- **Hundreds**: the number with its last two digits dropped (`3470xx`, `175xx`).
- **Window**: the first 3 digits for numbers of 6 or more digits, otherwise the first 2 (`347xxx`, `17xxx`).
- **Band**: length of the number and its first digit (`3xxxxx`).
- **Prefix**: the ticket type (`PC`, `C.A.`, Number only), for comparison.

**4. Families.** Passengers who share a ticket or a family (`FamilyId` from `families_v3.ipynb`) are linked into **parties**. A ticket grouping that only reflects a party sharing one fate is a party effect, not a ticket-number effect. Each test is therefore run twice: over passengers, and over parties, with each party counted once."""

setup = r'''import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 9.5,
    "text.color": INK, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "figure.dpi": 110,
})
RNG = np.random.default_rng(0)
PERMS = 4000

def find_data():
    from pathlib import Path
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "data" / "complete_data.csv").exists():
            return p / "data" / "complete_data.csv"
c = pd.read_csv(find_data())

# ---- non-ticket features
t = c["Name"].str.extract(r",\s*([^.]+)\.")[0].str.strip()
t = t.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
c["Title"] = t.where(t.isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")
c["OwnDeck"] = c["Cabin"].str[0].fillna("None").replace({"T": "A", "G": "F"})
c["LogFare"] = np.log1p(c["Fare"])
c["AgeMissing"] = c["Age"].isna().astype(int)
c["PclassS"] = c["Pclass"].astype(str)
c["Embarked"] = c["Embarked"].fillna("S")
CAT = ["PclassS", "Sex", "Title", "Embarked", "OwnDeck"]
NUM = ["Age", "AgeMissing", "SibSp", "Parch", "LogFare"]

# ---- ticket groupings, prefix ignored
def raw_prefix(x):
    parts = x.split(" ")
    if x.replace(" ", "").isdigit():
        return ""
    return " ".join(parts[:-1]) if parts[-1].isdigit() else x
c["Prefix"] = c["Ticket"].map(lambda x: "Number only" if raw_prefix(x) == "" else re.sub(r"[./ ]", "", raw_prefix(x)).upper())
c["Num"] = pd.to_numeric(c["Ticket"].str.extract(r"(\d+)$")[0])
s = c["Num"].map(lambda v: "" if pd.isna(v) else str(int(v)))
dig = s.str.len()
c["Band"] = np.where(dig > 0, s.str[:1] + dig.map(lambda d: "x" * max(d - 1, 0)), np.nan)
k = np.where(dig >= 6, 3, np.where(dig >= 4, 2, 1))
c["Window"] = [np.nan if d == 0 else v[:kk] + "x" * (d - kk) for v, d, kk in zip(s, dig, k)]
c["Hundreds"] = [np.nan if d == 0 else (v[:-2] + "xx" if d >= 4 else v[:1] + "x" * (d - 1)) for v, d in zip(s, dig)]

def runs(gap):
    u = np.sort(c["Num"].dropna().unique())
    rid = np.concatenate([[0], np.cumsum(np.diff(u) > gap)])
    first = pd.Series(u).groupby(rid).transform("min").astype(int).astype(str).values
    last = pd.Series(u).groupby(rid).transform("max").astype(int).astype(str).values
    lab = dict(zip(u, [f if f == l else f"{f}-{l[-3:] if len(l) > 3 else l}" for f, l in zip(first, last)]))
    return c["Num"].map(lab)
c["Run1"] = runs(1)
c["Run3"] = runs(3)

# ---- parties: connected components of shared ticket or shared family
parent = list(range(len(c)))
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]; i = parent[i]
    return i
for key in ["Ticket", "FamilyId"]:
    for _, idx in c.dropna(subset=[key]).groupby(key).indices.items():
        for j in idx[1:]:
            a, b = find(idx[0]), find(j)
            if a != b:
                parent[b] = a
c["Party"] = [find(i) for i in range(len(c))]

tr = c[c["Dataset"] == "train"].reset_index(drop=True)
y = tr["Survived"].astype(int).values
display(Markdown(
    f"{len(tr)} train passengers. {c['Party'].nunique():,} parties across train and test; "
    f"{int((tr.groupby('Party').size() > 1).sum())} parties have two or more members in train."
))'''

model_md = """## The model without ticket information"""

model = r'''pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM),
])
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0)
P = np.zeros((10, len(tr)))
for i, (a, b) in enumerate(cv.split(tr, y)):
    m = make_pipeline(pre, LogisticRegression(max_iter=5000)).fit(tr.iloc[a], y[a])
    P[i // 10, b] = m.predict_proba(tr.iloc[b])[:, 1]
tr["Pred"] = P.mean(axis=0)
tr["Resid"] = y - tr["Pred"]
spread = P.std(axis=0).mean()
display(Markdown(
    f"Held-out accuracy {accuracy_score(y, tr['Pred'] >= 0.5):.1%}, AUC {roc_auc_score(y, tr['Pred']):.3f}. "
    f"A passenger's 10 held-out predictions differ by {spread:.3f} on average (standard deviation), so averaging them "
    f"removes the luck of a single split. Residuals run from {tr['Resid'].min():.2f} to {tr['Resid'].max():.2f}; "
    f"their standard deviation is {tr['Resid'].std():.3f}."
))'''

level_md = """## Which groupings carry residual signal

For each grouping, the share of residual variation that lies between groups (eta squared), counting only groups with at least two train passengers. Even random groups explain some variation, especially small ones, so the observed value is compared with 4,000 random reshuffles of the residuals across the same passengers: **p** is the share of reshuffles that explain as much.

**By party** repeats the test with each party counted once (its mean residual), and a party assigned to the group most of its members fall in. If a grouping is significant over passengers but not over parties, the signal is parties sharing one fate rather than the ticket number. Exact ticket and number are not tested by party, because a ticket sits inside a single party by construction."""

level = r'''LEVELS = [("Exact ticket", "Ticket"), ("Number, any prefix", "Num"),
          ("Consecutive run, gap 1", "Run1"), ("Consecutive run, gap up to 3", "Run3"),
          ("Hundreds", "Hundreds"), ("Window", "Window"), ("Band", "Band"), ("Prefix", "Prefix")]

def perm_test(vals, labels, perms=PERMS):
    """eta squared over groups with 2+ members, against reshuffles; also each group's mean and permutation p."""
    lab = pd.Series(labels)
    ok = lab.notna() & lab.map(lab.value_counts()).ge(2)
    v = np.asarray(vals)[ok.values]
    codes, uniq = pd.factorize(lab[ok])
    n = np.bincount(codes)
    tot = ((v - v.mean()) ** 2).sum()
    def stat(x):
        mu = np.bincount(codes, weights=x) / n
        return (n * (mu - x.mean()) ** 2).sum() / tot, mu
    obs, mu = stat(v)
    null_eta = np.empty(perms)
    exceed = np.zeros(len(n))
    for p in range(perms):
        e, m_ = stat(RNG.permutation(v))
        null_eta[p] = e
        exceed += np.abs(m_ - v.mean()) >= np.abs(mu - v.mean()) - 1e-12
    groups = pd.DataFrame({"group": uniq, "n": n, "mean": mu, "p": (exceed + 1) / (perms + 1)})
    return dict(eta=obs, null=null_eta.mean(), p=(np.sum(null_eta >= obs) + 1) / (perms + 1),
                groups=len(n), covered=int(ok.sum())), groups

party = tr.groupby("Party").agg(Resid=("Resid", "mean"))
rows, GROUPS = [], {}
for name, col in LEVELS:
    r, g = perm_test(tr["Resid"].values, tr[col].values)
    GROUPS[name] = g
    row = {"Grouping": name, "Groups (2+ in train)": r["groups"], "Train passengers in them": r["covered"],
           "Eta squared": f"{r['eta']:.3f}", "Random groups": f"{r['null']:.3f}", "p": f"{r['p']:.4f}"}
    if col not in ("Ticket", "Num"):
        lab = tr.groupby("Party")[col].agg(lambda s_: s_.mode().iloc[0] if s_.notna().any() else np.nan)
        rp, _ = perm_test(party["Resid"].values, lab.reindex(party.index).values)
        row.update({"By party: groups": rp["groups"], "By party: eta squared": f"{rp['eta']:.3f}",
                    "By party: random": f"{rp['null']:.3f}", "By party: p": f"{rp['p']:.4f}"})
    rows.append(row)
LT = pd.DataFrame(rows).set_index("Grouping").fillna("")
display(LT)'''

pair_md = """## Residual correlation against ticket number distance

Every pair of train passengers on **different** tickets, grouped by how far apart their numbers are (prefix ignored), and the correlation of their residuals. Same ticket is shown at the top for comparison. Each distance is shown twice: all pairs, and pairs in different parties only. The grey band is the range of correlations from 1,000 random reshuffles of the residuals (95% of reshuffles fall inside it)."""

pair = r'''num = tr["Num"].values
z = ((tr["Resid"] - tr["Resid"].mean()) / tr["Resid"].std()).values
I, J = np.triu_indices(len(tr), 1)
same_ticket = tr["Ticket"].values[I] == tr["Ticket"].values[J]
same_party = tr["Party"].values[I] == tr["Party"].values[J]
dist = np.abs(num[I] - num[J])
BINS = [("Same ticket", same_ticket),
        ("Same number, other prefix", ~same_ticket & (dist == 0))]
for lo, hi, lab in [(1, 1, "1 apart"), (2, 3, "2 to 3 apart"), (4, 10, "4 to 10 apart"), (11, 30, "11 to 30 apart"),
                    (31, 100, "31 to 100 apart"), (101, 1000, "101 to 1,000 apart"), (1001, 10000, "1,001 to 10,000 apart")]:
    BINS.append((lab, ~same_ticket & (dist >= lo) & (dist <= hi)))
rows = []
nullp = 1000
perm_z = np.array([RNG.permutation(z) for _ in range(nullp)])
for lab, msk in BINS:
    for scope, m2 in [("All pairs", msk), ("Different parties", msk & ~same_party)]:
        if lab == "Same ticket" and scope == "Different parties":
            continue
        ii, jj = I[m2], J[m2]
        if len(ii) == 0:
            continue
        r = np.mean(z[ii] * z[jj])
        null = (perm_z[:, ii] * perm_z[:, jj]).mean(axis=1)
        rows.append(dict(bin=lab, scope=scope, pairs=len(ii), r=r, lo=np.quantile(null, 0.025), hi=np.quantile(null, 0.975),
                         p=(np.sum(np.abs(null) >= abs(r)) + 1) / (nullp + 1)))
PR = pd.DataFrame(rows)

order = [b for b, _ in BINS if b in PR["bin"].values and b != "Same number, other prefix"]
fig, ax = plt.subplots(figsize=(11, 0.5 * len(order) + 1.4))
for i, b in enumerate(order):
    for scope, off, col in [("All pairs", -0.14, BLUE), ("Different parties", 0.14, ORANGE)]:
        r = PR[(PR["bin"] == b) & (PR["scope"] == scope)]
        if r.empty:
            continue
        r = r.iloc[0]
        ax.plot([r["lo"], r["hi"]], [i + off, i + off], color=GRID, linewidth=6, solid_capstyle="butt", zorder=2)
        ax.scatter([r["r"]], [i + off], s=40, color=col, edgecolor=SURFACE, linewidth=1.2, zorder=4,
                   label=scope if i == 1 else None)
        ax.text(max(r["r"], r["hi"]) + 0.012, i + off, f"r = {r['r']:+.2f}   ({int(r['pairs']):,} pairs)",
                va="center", fontsize=7.5, color=INK2)
ax.axvline(0, color=AXIS, linewidth=1)
ax.set_yticks(range(len(order)), order); ax.invert_yaxis()
ax.set_xlim(-0.2, 0.65)
ax.grid(axis="x", color=GRID, linewidth=0.6); ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_xlabel("Correlation of the two passengers' residuals")
ax.legend(loc="lower right", frameon=False, fontsize=8.5)
ax.set_title("Do passengers with nearby ticket numbers share unexplained survival?", loc="left", fontsize=11, pad=10)
plt.show()
t_ = PR.assign(r=PR["r"].round(3), p=PR["p"].round(3))[["bin", "scope", "pairs", "r", "p"]]
t_.columns = ["Distance", "Scope", "Pairs", "Correlation", "p"]
display(t_.style.hide(axis="index"))'''

funnel_md = """## Which groups stand out

Each dot is one group with at least 3 train passengers, placed by its size and its mean residual. Random groups of the same size fall inside the grey funnel 95% of the time; dots outside it are labelled. With dozens of groups per panel, a few dots outside the funnel are expected by chance, so the table below adjusts for the number of groups tested (Benjamini-Hochberg q; below 0.10 is worth acting on)."""

funnel = r'''def bh(p):
    p = np.asarray(p); o = np.argsort(p); q = np.empty_like(p)
    q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.minimum(q, 1)

PANELS = ["Exact ticket", "Consecutive run, gap up to 3", "Hundreds", "Window"]
sd = tr["Resid"].std(); N = len(tr)
fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharey=True, gridspec_kw={"hspace": 0.38, "wspace": 0.06})
flag = []
for ax, name in zip(axes.flat, PANELS):
    g = GROUPS[name]; g = g[g["n"] >= 3].copy()
    g["q"] = bh(g["p"].values)
    g["level"] = name
    flag.append(g)
    nn = np.arange(3, max(g["n"].max(), 10) + 1)
    band = 1.96 * sd / np.sqrt(nn) * np.sqrt((N - nn) / (N - 1))
    ax.fill_between(nn, -band, band, color="#efeee9", zorder=1)
    out = np.abs(g["mean"]) > 1.96 * sd / np.sqrt(g["n"]) * np.sqrt((N - g["n"]) / (N - 1))
    ax.scatter(g.loc[~out, "n"], g.loc[~out, "mean"], s=16, color=MUTED, alpha=0.7, zorder=3)
    ax.scatter(g.loc[out, "n"], g.loc[out, "mean"], s=30, color=[BLUE if v > 0 else ORANGE for v in g.loc[out, "mean"]],
               edgecolor=SURFACE, linewidth=1, zorder=4)
    placed = []
    for _, r in g[out].sort_values(["n", "mean"]).iterrows():
        yl = r["mean"]
        while any(pn == r["n"] and abs(yl - py) < 0.045 for pn, py in placed):
            yl += 0.045 if r["mean"] > 0 else -0.045
        placed.append((r["n"], yl))
        ax.annotate(str(r["group"]).replace(".0", ""), (r["n"], r["mean"]), xytext=(r["n"] * 1.04, yl),
                    textcoords="data", fontsize=7, color=INK2, va="center")
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_xscale("log"); ax.set_xticks([3, 5, 10, 20, 50], ["3", "5", "10", "20", "50"]); ax.minorticks_off()
    ax.grid(axis="y", color=GRID, linewidth=0.6); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(name, loc="left", fontsize=10, pad=6)
    ax.set_xlabel("Train passengers in the group", fontsize=8.5)
for ax in axes[:, 0]:
    ax.set_ylabel("Mean residual (survived minus predicted)")
fig.suptitle("Ticket groups against the range expected by chance", x=0.125, ha="left", fontsize=11, y=0.95)
plt.show()

F = pd.concat(flag)
F = F[F["q"] < 0.25].sort_values("q")
detail = []
for _, r in F.iterrows():
    col = dict(LEVELS)[r["level"]]
    m = tr[tr[col].astype(str) == str(r["group"])]
    detail.append({"Grouping": r["level"], "Group": str(r["group"]).replace(".0", ""), "Train passengers": int(r["n"]),
                   "Parties": m["Party"].nunique(), "Class": "/".join(sorted(m["Pclass"].astype(str).unique())),
                   "Survived": f"{m['Survived'].mean():.0%}", "Model expected": f"{m['Pred'].mean():.0%}",
                   "Mean residual": f"{r['mean']:+.2f}", "p": f"{r['p']:.4f}", "q": f"{r['q']:.3f}"})
D = pd.DataFrame(detail)
if len(D):
    display(D.style.hide(axis="index"))
else:
    display(Markdown("No group has q below 0.25."))'''

nb = nbformat.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
md, code = nbformat.v4.new_markdown_cell, lambda s: nbformat.v4.new_code_cell(s, metadata={"jupyter": {"source_hidden": True}})
FF = lambda k: [md(FINDINGS[k])] if k in FINDINGS else []
nb.cells = ([md(intro), code(setup)] + FF("summary") + [md(model_md), code(model)] + FF("model") +
            [md(level_md), code(level)] + FF("level") + [md(pair_md), code(pair)] + FF("pair") +
            [md(funnel_md), code(funnel)] + FF("funnel"))
NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(AREA)}}).execute()
nbformat.write(nb, AREA / "tickets_residuals_v1.ipynb")
print("written", AREA / "tickets_residuals_v1.ipynb")
