"""Builds features/overview/features_v1.ipynb from data/train.csv.

The notebook holds text and chart images only (markdown cells, no code cells).
Re-run this script to regenerate it:

    python3 features/overview/build_features_v1.py
"""

import base64
import io
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MPath

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "train.csv"
OUT = Path(__file__).resolve().parent / "features_v1.ipynb"

# ---------------------------------------------------------------- style
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
C_DIED = "#eb6834"  # outcome 0
C_SURV = "#2a78d6"  # outcome 1, also the single hue for rate bars
DPI = 180
PX = DPI / 90  # device pixels per display pixel at the size the notebook shows
LABEL_BOX = dict(facecolor=SURFACE, edgecolor="none", pad=1.0)

plt.rcParams.update(
    {
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 9.5,
        "text.color": INK,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": MUTED,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def style_axes(ax, grid_axis="y"):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    base = "bottom" if grid_axis == "y" else "left"
    ax.spines[base].set_visible(True)
    ax.spines[base].set_linewidth(0.8)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)


def data_per_px(ax):
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform([(0, 0), (1, 1)])
    return abs(x1 - x0), abs(y1 - y0)


def rbar(ax, pos, length, thick, color, horizontal=False):
    """Bar from the zero baseline, square at the baseline, rounded at the data end."""
    if length <= 0:
        return
    dx, dy = data_per_px(ax)
    r = 4 * PX
    if horizontal:
        rl, rt = min(r * dx, length), min(r * dy, thick / 2)
        lo, hi = pos - thick / 2, pos + thick / 2
        verts = [
            (0, lo), (length - rl, lo), (length, lo), (length, lo + rt),
            (length, hi - rt), (length, hi), (length - rl, hi), (0, hi), (0, lo),
        ]
    else:
        rl, rt = min(r * dy, length), min(r * dx, thick / 2)
        lo, hi = pos - thick / 2, pos + thick / 2
        verts = [
            (lo, 0), (lo, length - rl), (lo, length), (lo + rt, length),
            (hi - rt, length), (hi, length), (hi, length - rl), (hi, 0), (lo, 0),
        ]
    codes = [
        MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3,
        MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", zorder=3))


def to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def outcome_legend(ax):
    ax.legend(
        handles=[
            Patch(facecolor=C_DIED, label="Did not survive (0)"),
            Patch(facecolor=C_SURV, label="Survived (1)"),
        ],
        loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False,
        fontsize=8.5, handlelength=1.0, handleheight=1.0, columnspacing=1.4,
        borderaxespad=0.2, labelcolor=INK2,
    )


# ---------------------------------------------------------------- data
df = pd.read_csv(DATA)
N = len(df)
SURV = int(df.Survived.sum())
OVERALL = df.Survived.mean()


def summarise(groups, order=None):
    """n, died, survived and survival rate per group."""
    t = df.groupby(groups, dropna=False, observed=False).Survived.agg(n="size", surv="sum")
    if order is not None:
        t = t.reindex(order)
    t = t.fillna(0).astype(int)
    t["died"] = t.n - t.surv
    t["rate"] = t.surv / t.n
    return t


# ---------------------------------------------------------------- charts
def two_panel(t, labels, xlabel, width):
    """Left: passengers by outcome. Right: survival rate on a 0 to 1 axis."""
    k = len(t)
    fig = plt.figure(figsize=(width, 3.9), dpi=DPI)
    gs = fig.add_gridspec(1, 2, left=0.06, right=0.95, top=0.82, bottom=0.24, wspace=0.2)
    x = np.arange(k)

    # counts
    ax = fig.add_subplot(gs[0])
    style_axes(ax)
    top = max(t.died.max(), t.surv.max())
    ax.set_xlim(-0.6, k - 0.4)
    ax.set_ylim(0, top * 1.14)
    dx, _ = data_per_px(ax)
    thick = min(0.36, 24 * PX * dx)
    off = thick / 2 + PX * dx
    for i, (d, s) in enumerate(zip(t.died, t.surv)):
        rbar(ax, i - off, d, thick, C_DIED)
        rbar(ax, i + off, s, thick, C_SURV)
        for xpos, v in ((i - off, d), (i + off, s)):
            ax.text(xpos, v + top * 0.015, f"{v}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, labels)
    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_title("Passengers", loc="left", fontsize=10.5, fontweight="bold", pad=22)
    outcome_legend(ax)

    # rate
    ax = fig.add_subplot(gs[1])
    style_axes(ax)
    ax.set_xlim(-0.6, k - 0.4)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.50", "0.75", "1"])
    dx, _ = data_per_px(ax)
    thick = min(0.5, 24 * PX * dx)
    ax.axhline(OVERALL, color=MUTED, linewidth=0.9, zorder=2)
    ax.text(1.012, OVERALL, f"all\n{OVERALL:.2f}", transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=8, color=MUTED, linespacing=1.1)
    for i, r in enumerate(t.rate):
        rbar(ax, i, r, thick, C_SURV)
        ax.text(i, r + 0.015, f"{r:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK,
                bbox=LABEL_BOX, zorder=4)
    ax.set_xticks(x, [f"{lab}\nn={n}" for lab, n in zip(labels, t.n)])
    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_title("Survival rate (0 to 1)", loc="left", fontsize=10.5, fontweight="bold", pad=22)
    return to_png(fig)


def passenger_id_chart(bands):
    fig = plt.figure(figsize=(11.5, 5.6), dpi=DPI)
    gs = fig.add_gridspec(2, 1, left=0.12, right=0.95, top=0.9, bottom=0.1,
                          hspace=0.5, height_ratios=[1, 1.15])

    ax = fig.add_subplot(gs[0])
    style_axes(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.17, 0.17, N)
    for outcome, colour in ((0, C_DIED), (1, C_SURV)):
        m = (df.Survived == outcome).to_numpy()
        ax.scatter(df.PassengerId[m], df.Survived[m] + jitter[m], s=13, color=colour,
                   alpha=0.75, linewidths=0, zorder=3)
    ax.set_xlim(-10, 901)
    ax.set_ylim(-0.42, 1.42)
    ax.set_yticks([0, 1], [f"0  did not survive\nn={N - SURV}", f"1  survived\nn={SURV}"])
    ax.tick_params(axis="y", labelcolor=INK2)
    ax.set_xticks(range(0, 901, 100))
    ax.set_title("Each passenger: survived (1) or did not survive (0)", loc="left",
                 fontsize=10.5, fontweight="bold", pad=10)

    ax = fig.add_subplot(gs[1])
    style_axes(ax)
    ax.set_xlim(-10, 901)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.50", "0.75", "1"])
    dx, _ = data_per_px(ax)
    thick = min(60, 24 * PX * dx)
    ax.axhline(OVERALL, color=MUTED, linewidth=0.9, zorder=2)
    ax.text(1.008, OVERALL, f"all\n{OVERALL:.2f}", transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=8, color=MUTED, linespacing=1.1)
    for (lo, hi), r in zip(bands.index, bands.rate):
        mid = (lo + hi) / 2
        rbar(ax, mid, r, thick, C_SURV)
        ax.text(mid, r + 0.015, f"{r:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK,
                bbox=LABEL_BOX, zorder=4)
    ax.set_xticks(range(0, 901, 100))
    ax.set_xlabel("PassengerId", labelpad=8)
    ax.set_title("Survival rate (0 to 1) per block of 100 IDs", loc="left",
                 fontsize=10.5, fontweight="bold", pad=10)
    return to_png(fig)


def ticket_detail_chart(t, labels):
    k = len(t)
    fig = plt.figure(figsize=(9.5, 0.3 * k + 1.3), dpi=DPI)
    ax = fig.add_axes([0.2, 0.07, 0.68, 0.86])
    style_axes(ax, grid_axis="x")
    ax.set_ylim(k - 0.4, -0.6)
    ax.set_xlim(0, 1.1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.50", "0.75", "1"])
    ax.tick_params(axis="x", labelcolor=MUTED, top=True, labeltop=True, labelbottom=False)
    _, dy = data_per_px(ax)
    thick = min(0.55, 24 * PX * dy)
    ax.axvline(OVERALL, color=MUTED, linewidth=0.9, zorder=2)
    ax.text(OVERALL, k - 0.25, f"all {OVERALL:.2f}", ha="center", va="top", fontsize=8, color=MUTED)
    for i, (r, s, n) in enumerate(zip(t.rate, t.surv, t.n)):
        rbar(ax, i, r, thick, C_SURV, horizontal=True)
        ax.text(r + 0.012, i, f"{r:.2f}", ha="left", va="center", fontsize=8.5, color=INK,
                bbox=LABEL_BOX, zorder=4)
        ax.text(1.02, i, f"{s} of {n}", transform=ax.get_yaxis_transform(), ha="left",
                va="center", fontsize=8.5, color=INK2)
    ax.text(1.02, -0.95, "survived", transform=ax.get_yaxis_transform(), ha="left",
            va="center", fontsize=8.5, color=MUTED)
    ax.set_yticks(range(k), labels)
    ax.tick_params(axis="y", labelcolor=INK2)
    ax.set_title("Survival rate (0 to 1) for every ticket type", loc="left",
                 fontsize=10.5, fontweight="bold", pad=26)
    return to_png(fig)


# ---------------------------------------------------------------- notebook helpers
def md_table(t, labels, first_col):
    rows = [f"| {first_col} | Passengers | Did not survive | Survived | Survival rate |",
            "|---|---:|---:|---:|---:|"]
    for lab, (_, r) in zip(labels, t.iterrows()):
        rows.append(f"| {lab} | {int(r.n)} | {int(r.died)} | {int(r.surv)} | {r.rate:.2f} |")
    return "\n".join(rows)


def flat(label):
    return label.replace("\n", " ")


cells = []


def add(text, images=None):
    cell = nbformat.v4.new_markdown_cell(text.strip())
    if images:
        cell["attachments"] = {name: {"image/png": b64} for name, b64 in images.items()}
    cells.append(cell)


# ---------------------------------------------------------------- intro
add(f"""
# Titanic features v1

Survival against one feature at a time, from train.csv. {N} passengers, {SURV} survived, so the survival rate for all passengers is {OVERALL:.2f}. That rate is the grey line marked "all" on every rate chart.

**How to read the charts**

- Left panel: the number of passengers in each group, split into did not survive (0) and survived (1)
- Right panel: the survival rate of each group on a 0 to 1 axis. A rate of 0.74 means 74 of every 100 passengers in the group survived
- n is the number of passengers in the group. One passenger moves the rate of a small group a long way, so read n before the rate

test.csv is not used here because it has no Survived column.
""")

# ---------------------------------------------------------------- 1 PassengerId
edges = [0, 100, 200, 300, 400, 500, 600, 700, 800, 891]
band = pd.cut(df.PassengerId, edges)
bands = summarise(band)
bands.index = pd.Index([(int(iv.left) + 1, int(iv.right)) for iv in bands.index])
band_labels = [f"{lo} to {hi}" for lo, hi in bands.index]
se = (OVERALL * (1 - OVERALL) / 100) ** 0.5
add(f"""
## 1. PassengerId

PassengerId is a row number from 1 to {N}. In the top chart each dot is one passenger, placed at 1 if they survived and 0 if they did not. The dots are spread a little above and below 0 and 1 so that they do not sit on top of each other. The bottom chart is the survival rate of each block of 100 IDs.

![Survival by PassengerId](attachment:passenger_id.png)

Survivors and non-survivors are spread evenly along the whole ID range. The rate per block of 100 moves between {bands.rate.min():.2f} and {bands.rate.max():.2f} with no trend. In 19 cases out of 20, a block of 100 passengers drawn at random from a population with a rate of {OVERALL:.2f} lands between about {OVERALL - 1.96 * se:.2f} and {OVERALL + 1.96 * se:.2f}, so this is close to what chance alone produces.

{md_table(bands, band_labels, "PassengerId")}
""", {"passenger_id.png": passenger_id_chart(bands)})

# ---------------------------------------------------------------- 2 Sex
t = summarise(df.Sex, ["female", "male"])
labels = ["female", "male"]
add(f"""
## 2. Sex

![Survival by sex](attachment:sex.png)

Women survived at {t.rate["female"]:.2f} and men at {t.rate["male"]:.2f}. Both groups are large, so the gap is not a small-sample effect.

{md_table(t, labels, "Sex")}
""", {"sex.png": two_panel(t, labels, "Sex", 8.6)})

# ---------------------------------------------------------------- 3 SibSp
t = summarise(df.SibSp)
labels = [str(v) for v in t.index]
high = t[t.index >= 3]
add(f"""
## 3. SibSp

SibSp is the number of siblings and spouses travelling with the passenger.

![Survival by SibSp](attachment:sibsp.png)

Passengers with 1 sibling or spouse aboard survived at {t.rate[1]:.2f}, against {t.rate[0]:.2f} for those with none. From 3 upward the rate falls, and none of the {int(t.n[5] + t.n[8])} passengers with 5 or 8 survived. Only {int(high.n.sum())} passengers have 3 or more, so those rates rest on small groups.

{md_table(t, labels, "SibSp")}
""", {"sibsp.png": two_panel(t, labels, "SibSp: siblings and spouses aboard", 11.5)})

# ---------------------------------------------------------------- 4 Parch
t = summarise(df.Parch)
labels = [str(v) for v in t.index]
high = t[t.index >= 3]
add(f"""
## 4. Parch

Parch is the number of parents and children travelling with the passenger.

![Survival by Parch](attachment:parch.png)

The shape matches SibSp: {t.rate[0]:.2f} with none, {t.rate[1]:.2f} with 1 and {t.rate[2]:.2f} with 2. Only {int(high.n.sum())} passengers have 3 or more, which is too few to read a rate from.

{md_table(t, labels, "Parch")}
""", {"parch.png": two_panel(t, labels, "Parch: parents and children aboard", 11.5)})

# ---------------------------------------------------------------- 5 Ticket type
def raw_prefix(ticket):
    parts = ticket.split(" ")
    if ticket.replace(" ", "").isdigit():
        return ""
    return " ".join(parts[:-1]) if parts[-1].isdigit() else ticket


def ticket_key(ticket):
    prefix = raw_prefix(ticket)
    return "NUMBER" if prefix == "" else re.sub(r"[./ ]", "", prefix).upper()


df["TicketPrefix"] = df.Ticket.map(raw_prefix)
df["TicketType"] = df.Ticket.map(ticket_key)
spellings = (
    df[df.TicketType != "NUMBER"].groupby("TicketType").TicketPrefix
    .agg(lambda s: list(s.value_counts().index))
)
name = {key: sp[0] for key, sp in spellings.items()}
name["NUMBER"] = "Number only"

full = summarise(df.TicketType)
full["label"] = [name[key] for key in full.index]
full = full.sort_values(["n", "label"], ascending=[False, True])
big = full[full.n >= 10]
small = full[full.n < 10]
other = pd.DataFrame(
    {"n": [small.n.sum()], "surv": [small.surv.sum()], "died": [small.died.sum()],
     "rate": [small.surv.sum() / small.n.sum()], "label": ["Other\nprefixes"]},
    index=["OTHER"],
)
head = pd.concat([big, other])
head_labels = [
    lab.replace("Number only", "Number\nonly") if "/" not in lab or len(lab) < 7
    else lab.replace("/", "/\n", 1)
    for lab in head.label
]

prefixed = df[df.TicketType != "NUMBER"]
num = full.loc["NUMBER"]
low = big[(big.rate < 0.15)]
table_rows = ["| Ticket type | Spellings merged | Passengers | Did not survive | Survived | Survival rate |",
              "|---|---|---:|---:|---:|---:|"]
for key, r in full.iterrows():
    merged = "" if key == "NUMBER" else ", ".join(f"`{s}`" for s in spellings[key])
    table_rows.append(f"| {r.label} | {merged} | {int(r.n)} | {int(r.died)} | {int(r.surv)} | {r.rate:.2f} |")

add(f"""
## 5. Ticket type

A ticket is either a number alone ({int(num.n)} of {N} passengers) or a letter prefix followed by a number. Spellings of the same prefix are merged by ignoring capitals, full stops, slashes and spaces, so `A/5`, `A/5.`, `A./5.` and `A.5.` are one type. That leaves {len(full)} types. 4 tickets read `LINE` with no number and are kept as their own type.

The first chart shows the {len(big)} types with 10 or more passengers and folds the other {len(small)} into "Other prefixes". The second chart shows every type.

![Survival by ticket type](attachment:ticket_type.png)

![Survival rate for every ticket type](attachment:ticket_type_all.png)

Number-only tickets and prefixed tickets survive at the same rate: {num.rate:.2f} against {prefixed.Survived.mean():.2f} for the {len(prefixed)} passengers with any prefix. The prefixes differ from each other. PC is at {full.rate["PC"]:.2f} on {int(full.n["PC"])} passengers, while {", ".join(low.label[:-1])} and {low.label.iloc[-1]} are all under 0.15. {len(small)} of the {len(full)} types have fewer than 10 passengers, and {int((full.n == 1).sum())} have only 1.

{chr(10).join(table_rows)}
""", {
    "ticket_type.png": two_panel(head, head_labels, "Ticket type", 12.5),
    "ticket_type_all.png": ticket_detail_chart(full, [f"{lab}   n={n}" for lab, n in zip(full.label, full.n)]),
})

# ---------------------------------------------------------------- 6 Cabin letter
letter = df.Cabin.str[0].fillna("None")
order = sorted(x for x in letter.unique() if x != "None") + ["None"]
t = summarise(letter, order)
labels = [x if x != "None" else "None\nrecorded" for x in order]
has = df[df.Cabin.notna()]
multi_letter = int(has.Cabin.map(lambda c: len({p[0] for p in c.split()}) > 1).sum())
top3 = t.drop("None")
top3 = top3[top3.n >= 10].sort_values("rate", ascending=False).head(3)
add(f"""
## 6. Cabin by letter

The cabin letter is the first letter of the Cabin field, which is the deck. {int(t.n["None"])} of {N} passengers have no cabin recorded. Where a passenger has several cabins the first one is used; {multi_letter} entries such as `F G73` list two letters and are counted under the first.

![Survival by cabin letter](attachment:cabin.png)

Passengers with a cabin recorded survived at {has.Survived.mean():.2f} ({int(has.Survived.sum())} of {len(has)}), against {t.rate["None"]:.2f} for those without. Among the letters, {", ".join(sorted(top3.index)[:-1])} and {sorted(top3.index)[-1]} are highest, at {top3.rate.min():.2f} to {top3.rate.max():.2f}. G and T have {int(t.n["G"])} passengers and {int(t.n["T"])} passenger.

{md_table(t, [flat(x) for x in labels], "Cabin letter")}
""", {"cabin.png": two_panel(t, labels, "Cabin letter (deck)", 12.5)})

# ---------------------------------------------------------------- 7 Embarked
port = df.Embarked.fillna("None")
order = ["C", "Q", "S", "None"]
t = summarise(port, order)
labels = ["C\nCherbourg", "Q\nQueenstown", "S\nSouthampton", "Not\nrecorded"]
add(f"""
## 7. Embarked by letter

Embarked is the port where the passenger boarded: C is Cherbourg, Q is Queenstown and S is Southampton. {int(t.n["None"])} passengers have no port recorded, and both survived.

![Survival by port of embarkation](attachment:embarked.png)

Cherbourg is at {t.rate["C"]:.2f}, Queenstown at {t.rate["Q"]:.2f} and Southampton at {t.rate["S"]:.2f}. Southampton has {int(t.n["S"])} of the {N} passengers.

{md_table(t, [flat(x) for x in labels], "Embarked")}
""", {"embarked.png": two_panel(t, labels, "Embarked", 10.5)})

# ---------------------------------------------------------------- 8 Fare
fare_edges = [7.5, 10, 15, 25, 35, 60, 100]
fare_labels = ["0\n(free)", "under\n7.5"] + [
    f"{a:g} to\n{b:g}" for a, b in zip(fare_edges[:-1], fare_edges[1:])
] + ["100\nand over"]


def fare_window(fare):
    if fare == 0:
        return 0
    return 1 + int(np.searchsorted(fare_edges, fare, side="right"))


t = summarise(df.Fare.map(fare_window), list(range(len(fare_labels))))
under15 = int((df.Fare < 15).sum())
steps_down = [
    (flat(fare_labels[i - 1]), flat(fare_labels[i]), t.rate[i - 1], t.rate[i])
    for i in range(1, len(t)) if t.rate[i] < t.rate[i - 1]
]
assert len(steps_down) == 1, steps_down
prev_lab, this_lab, prev_rate, this_rate = steps_down[0]
add(f"""
## 8. Fare by window

Fare is the price of the ticket in GBP. The windows are narrow where fares are bunched and wide where they are sparse: {under15} of the {N} passengers paid under 15, and only {int(t.n.iloc[-1])} paid 100 or more. Each window includes its lower bound and excludes its upper bound, so a fare of exactly 10 is in "10 to 15". The {int(t.n[0])} passengers with a fare of 0 are their own window.

![Survival by fare window](attachment:fare.png)

The survival rate rises with the fare, from {t.rate[0]:.2f} for the free tickets to {t.rate.iloc[-1]:.2f} at 100 and over. The one step down is {this_lab} at {this_rate:.2f}, after {prev_lab} at {prev_rate:.2f}.

{md_table(t, [flat(x) for x in fare_labels], "Fare (GBP)")}
""", {"fare.png": two_panel(t, fare_labels, "Fare window (GBP)", 12.5)})

# ---------------------------------------------------------------- 9 Pclass
t = summarise(df.Pclass, [1, 2, 3])
labels = ["1\nfirst", "2\nsecond", "3\nthird"]
add(f"""
## 9. Pclass

Pclass is the ticket class: 1 is first class, 2 is second and 3 is third.

![Survival by Pclass](attachment:pclass.png)

First class survived at {t.rate[1]:.2f}, second at {t.rate[2]:.2f} and third at {t.rate[3]:.2f}. Third class has {int(t.n[3])} of the {N} passengers, and {int(t.died[3])} of the {N - SURV} who did not survive.

{md_table(t, [flat(x) for x in labels], "Pclass")}
""", {"pclass.png": two_panel(t, labels, "Pclass: ticket class", 9.6)})

# ---------------------------------------------------------------- write
nb = nbformat.v4.new_notebook(cells=cells)
nbformat.validate(nb)
OUT.parent.mkdir(exist_ok=True)
nbformat.write(nb, OUT)
print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB, {len(cells)} cells)")
