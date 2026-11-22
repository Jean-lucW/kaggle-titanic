"""Family matching rules for the Titanic passenger list.

Step 1, relationship filter (fixed logic from SibSp and Parch, nuclear families):
    For a passenger with SibSp = x and Parch = y:
    - a sibling has SibSp = x and Parch = y
    - a spouse has SibSp >= 1 and the same Parch (the same children)
    - a parent has Parch = x + 1 and, when y = 2, SibSp >= 1 (the other parent)
    - a child is the reverse of a parent
Step 2, feature matches between every pair that passes step 1:
    surname (or maiden name in brackets, matched to a surname or another maiden name;
    a hyphenated surname matches either of its parts, so Kink matches Kink-Heilmann),
    ticket, cabin, class, port, fare.
    More matches means a stronger link.
Step 3, build families: join the strongest pairs first. A join is allowed only when
    - the new family is no bigger than SibSp + Parch + 1 for anyone in it, and
    - every pair inside it passes step 1.
"""
from itertools import combinations
from pathlib import Path

import pandas as pd

FEATURES = ["Surname", "Ticket", "Cabin", "Pclass", "Embarked", "Fare"]
STRONG = ["Surname", "Ticket", "Cabin"]   # a link needs at least one of these


def data_dir():
    here = Path.cwd()
    for p in [here, *here.parents]:
        if (p / "data" / "train.csv").exists():
            return p / "data"
    raise FileNotFoundError("data/train.csv not found above " + str(here))


def load():
    d = data_dir()
    train, test = pd.read_csv(d / "train.csv"), pd.read_csv(d / "test.csv")
    p = pd.concat([train, test], ignore_index=True).sort_values("PassengerId", ignore_index=True)
    p["Surname"] = p["Name"].str.split(",").str[0].str.strip()
    # Maiden name: last word inside brackets, e.g. "Hogeboom, Mrs. John C (Anna Andrews)" -> Andrews
    p["Maiden"] = p["Name"].str.extract(r"\(([^)]*)\)", expand=False).str.replace('"', "").str.split().str[-1]
    p["Cabins"] = p["Cabin"].fillna("").str.split().apply(set)
    p["Expected"] = p["SibSp"] + p["Parch"] + 1
    return train, test, p


def relation(a, b):
    """Possible roles of b relative to a, from SibSp and Parch alone. Empty means impossible."""
    roles = []
    if a.SibSp >= 1 and a.SibSp == b.SibSp and a.Parch == b.Parch:
        roles.append("sibling")
    if a.SibSp >= 1 and b.SibSp >= 1 and a.Parch == b.Parch:
        roles.append("spouse")
    if a.Parch >= 1 and b.Parch == a.SibSp + 1 and b.SibSp >= a.Parch - 1:
        roles.append("parent")
    if b.Parch >= 1 and a.Parch == b.SibSp + 1 and a.SibSp >= b.Parch - 1:
        roles.append("child")
    return roles


def matches(a, b):
    """Which features a and b share."""
    return {
        "Surname": (a.Surname == b.Surname or a.Maiden == b.Surname or b.Maiden == a.Surname
                    or (isinstance(a.Maiden, str) and a.Maiden == b.Maiden)
                    or bool(set(a.Surname.split("-")) & set(b.Surname.split("-")))),
        "Ticket": a.Ticket == b.Ticket,
        "Cabin": bool(a.Cabins & b.Cabins),
        "Pclass": a.Pclass == b.Pclass,
        "Embarked": a.Embarked == b.Embarked,
        "Fare": a.Fare == b.Fare,
    }


def candidates(p):
    """Every pair of passengers with family aboard that passes the SibSp/Parch filter."""
    fam = p[p["Expected"] > 1]
    rows = []
    for (i, a), (j, b) in combinations(fam.iterrows(), 2):
        roles = relation(a, b)
        if not roles:
            continue
        m = matches(a, b)
        rows.append({"i": i, "j": j, "A": a.PassengerId, "B": b.PassengerId,
                     "Role of B": " or ".join(roles), **m,
                     "Score": sum(m.values()), "Strong": any(m[k] for k in STRONG)})
    return pd.DataFrame(rows)


def build(p, cand):
    """Join the strongest compatible pairs into families. Returns a group label per row."""
    fam = p[p["Expected"] > 1]
    ok = set(zip(cand["i"], cand["j"])) | set(zip(cand["j"], cand["i"]))
    members = {i: {i} for i in fam.index}
    cap = fam["Expected"].to_dict()
    root = {i: i for i in fam.index}
    links = cand[cand["Strong"]].sort_values(["Score", "Ticket", "Surname"], ascending=False)
    for i, j in zip(links["i"], links["j"]):
        ri, rj = root[i], root[j]
        if ri == rj:
            continue
        joined = members[ri] | members[rj]
        if len(joined) > min(cap[k] for k in joined):
            continue
        if not all((a, b) in ok for a in members[ri] for b in members[rj]):
            continue
        members[ri] = joined
        for k in members.pop(rj):
            root[k] = ri
    return pd.Series(root)


def check(p, label):
    """Correct when the family size equals SibSp + Parch + 1 for every member."""
    fam = p.loc[label.index].assign(G=label)
    g = fam.groupby("G")["Expected"].agg(["size", "min", "max"])
    good = (g["min"] == g["max"]) & (g["size"] == g["min"])
    size = fam.groupby("G")["G"].transform("size")
    out = pd.Series("Incomplete family", index=fam.index)
    out[fam["G"].map(good)] = "Complete family"
    out[size == 1] = "No match found"
    return out


# ---------------------------------------------------------------------------
# v3: households and the per-person check
#
# A household is a set of passengers linked by surname, maiden name, ticket or cabin.
# Inside a household every pair is labelled sibling, spouse, parent, child or none
# (none = in the household but not a direct relative, e.g. a brother's wife).
# A person is FULLY PLACED when, in the best labelling, the household holds exactly
#     SibSp people labelled sibling or spouse, and
#     Parch people labelled parent or child.
# Labelling rules:
#     - at most one spouse, of the opposite sex, a man (not Master) and a Mrs with
#       his surname; when her name is "Mrs. <first name> ..." his first name must match
#     - at most two parents; a Master is never a parent; a parent is at least 14 years
#       older than the child when both ages are known
#     - siblings of siblings are siblings, and siblings share the same parents
#     - everyone in a household has at least one direct relative in it, and the
#       direct relations connect the whole household
# ---------------------------------------------------------------------------

MALE_ADULT = {"Mr", "Dr", "Rev", "Col", "Major", "Capt", "Sir", "Don", "Jonkheer"}


def add_person_fields(p):
    p = p.copy()
    p["Title"] = p["Name"].str.extract(r",\s*([^.]+)\.", expand=False).str.strip()
    rest = p["Name"].str.split(".", n=1).str[1].str.replace(r"\(.*\)", "", regex=True).str.strip()
    p["First"] = rest.str.split().str[0].str.replace('"', "")
    return p


def _spouse_ok(a, b):
    if a.Sex == b.Sex or a.SibSp < 1 or b.SibSp < 1:
        return False
    man, woman = (a, b) if a.Sex == "male" else (b, a)
    if man.Title not in MALE_ADULT or woman.Title not in {"Mrs", "Mme", "Lady", "Countess", "Dona"}:
        return False
    if woman.Surname != man.Surname:
        return False      # a wife carries her husband's surname
    if woman.Title == "Mrs" and isinstance(woman.First, str) and woman.First != man.First:
        return False      # "Mrs. Wilhelm ..." is married to a Wilhelm
    return True


def _parent_ok(par, ch):
    if par.Parch < 1 or ch.Parch < 1 or par.Title == "Master":
        return False
    if pd.notna(par.Age) and pd.notna(ch.Age) and par.Age - ch.Age < 14:
        return False
    if pd.notna(par.Age) and par.Age < 14:
        return False
    return True


def allowed(a, b):
    """Labels for the pair (a, b), read as 'b is a's ...'."""
    out = ["none"]
    if a.SibSp >= 1 and b.SibSp >= 1:
        out.append("sibling")
        if _spouse_ok(a, b):
            out.append("spouse")
    if _parent_ok(b, a):
        out.append("parent")
    if _parent_ok(a, b):
        out.append("child")
    return out


FLIP = {"none": "none", "sibling": "sibling", "spouse": "spouse", "parent": "child", "child": "parent"}


def best_labelling(p, members, require_exact=False, max_nodes=300_000):
    """Search the pair labels for a household.

    Returns (number of people fully placed, labels dict, valid) where valid means a
    connected labelling within everyone's SibSp and Parch exists.
    """
    m = sorted(members)
    rows = {i: p.loc[i] for i in m}
    pairs = [(a, b) for k, a in enumerate(m) for b in m[k + 1:]]
    dom = {(a, b): allowed(rows[a], rows[b]) for a, b in pairs}
    need_s = {i: rows[i].SibSp for i in m}
    need_p = {i: rows[i].Parch for i in m}
    cnt_s = dict.fromkeys(m, 0); cnt_p = dict.fromkeys(m, 0)
    spouse = dict.fromkeys(m, 0); parents = dict.fromkeys(m, 0)
    # remaining pairs per person that could still add to each count
    left_s = dict.fromkeys(m, 0); left_p = dict.fromkeys(m, 0)
    for (a, b), d in dom.items():
        if "sibling" in d or "spouse" in d:
            left_s[a] += 1; left_s[b] += 1
        if "parent" in d or "child" in d:
            left_p[a] += 1; left_p[b] += 1
    lab = {}
    best = [-1, None]
    nodes = [0]

    def sat(i):
        return cnt_s[i] == need_s[i] and cnt_p[i] == need_p[i]

    def leaf_ok():
        rel = {i: set() for i in m}
        for (a, b), l in lab.items():
            if l != "none":
                rel[a].add(b); rel[b].add(a)
        seen, stack = {m[0]}, [m[0]]
        while stack:
            for nb in rel[stack.pop()]:
                if nb not in seen:
                    seen.add(nb); stack.append(nb)
        if len(seen) != len(m):
            return False
        par = {i: frozenset(b if lab[(i, b) if i < b else (b, i)] == ("parent" if i < b else "child") else None
                            for b in m if b != i) - {None} for i in m}
        for (a, b), l in lab.items():
            if l == "sibling" and par[a] != par[b]:
                return False
        return True

    def get(a, b):
        return lab.get((a, b) if a < b else (b, a))

    def dfs(k):
        nodes[0] += 1
        if nodes[0] > max_nodes:
            return
        possible = sum(1 for i in m if cnt_s[i] + left_s[i] >= need_s[i] and cnt_p[i] + left_p[i] >= need_p[i])
        if possible <= best[0]:
            return
        if require_exact and possible < len(m):
            return
        if k == len(pairs):
            if leaf_ok():
                best[0] = sum(sat(i) for i in m); best[1] = dict(lab)
            return
        a, b = pairs[k]
        d = dom[(a, b)]
        cs, cp = ("sibling" in d or "spouse" in d), ("parent" in d or "child" in d)
        if cs: left_s[a] -= 1; left_s[b] -= 1
        if cp: left_p[a] -= 1; left_p[b] -= 1
        for l in reversed(d):
            ds = l in ("sibling", "spouse"); dp = l in ("parent", "child")
            if ds and (cnt_s[a] >= need_s[a] or cnt_s[b] >= need_s[b]):
                continue
            if dp and (cnt_p[a] >= need_p[a] or cnt_p[b] >= need_p[b]):
                continue
            if l == "spouse" and (spouse[a] or spouse[b]):
                continue
            if l == "parent" and parents[a] >= 2:
                continue
            if l == "child" and parents[b] >= 2:
                continue
            # sibling transitivity against already-labelled pairs
            bad = False
            for c in m:
                if c in (a, b):
                    continue
                lac, lbc = get(a, c), get(b, c)
                if lac is None or lbc is None:
                    continue
                if (l == "sibling") != (lac == "sibling" and lbc == "sibling") and \
                        (l == "sibling" and (lac == "sibling") != (lbc == "sibling") or
                         l != "sibling" and lac == "sibling" and lbc == "sibling"):
                    bad = True; break
            if bad:
                continue
            lab[(a, b)] = l
            if ds: cnt_s[a] += 1; cnt_s[b] += 1
            if dp: cnt_p[a] += 1; cnt_p[b] += 1
            if l == "spouse": spouse[a] += 1; spouse[b] += 1
            if l == "parent": parents[a] += 1
            if l == "child": parents[b] += 1
            dfs(k + 1)
            if ds: cnt_s[a] -= 1; cnt_s[b] -= 1
            if dp: cnt_p[a] -= 1; cnt_p[b] -= 1
            if l == "spouse": spouse[a] -= 1; spouse[b] -= 1
            if l == "parent": parents[a] -= 1
            if l == "child": parents[b] -= 1
            del lab[(a, b)]
            if best[0] == len(m):
                break
        if cs: left_s[a] += 1; left_s[b] += 1
        if cp: left_p[a] += 1; left_p[b] += 1

    if len(m) == 1:
        return 0, {}, True
    dfs(0)
    return best[0], best[1], best[1] is not None


def nuclear(a, b, label):
    """True when the label also fits the closed-family rule from v2."""
    if label == "sibling":
        return a.SibSp == b.SibSp and a.Parch == b.Parch
    if label == "spouse":
        return a.Parch == b.Parch
    if label == "parent":
        return b.Parch == a.SibSp + 1
    if label == "child":
        return a.Parch == b.SibSp + 1
    return False


def build_households(p, cand):
    """Join the strongest links first. A link joins two households when the two
    passengers can be given a direct relation (sibling, spouse, parent or child)
    that neither of them has used up: the sibling-or-spouse count stays within SibSp,
    the parent-or-child count within Parch, one spouse, two parents at most.
    Labels that also fit the closed-family rule are tried first. Links inside a
    household are labelled too, so every relation found uses up the counts.
    Returns the household of each passenger, the households, and the labels."""
    fam = p[p["Expected"] > 1]
    members = {i: {i} for i in fam.index}
    root = {i: i for i in fam.index}
    cs = dict.fromkeys(fam.index, 0); cp = dict.fromkeys(fam.index, 0)
    sp = dict.fromkeys(fam.index, 0); pa = dict.fromkeys(fam.index, 0)
    labels = {}
    links = cand[cand["Strong"]].sort_values(["Score", "Ticket", "Surname"], ascending=False)
    for i, j in zip(links["i"], links["j"]):
        ri, rj = root[i], root[j]
        a, b = p.loc[i], p.loc[j]
        opts = [l for l in allowed(a, b) if l != "none"]
        opts.sort(key=lambda l: (not nuclear(a, b, l), l != "spouse"))
        for l in opts:
            if l in ("sibling", "spouse") and (cs[i] >= a.SibSp or cs[j] >= b.SibSp):
                continue
            if l in ("parent", "child") and (cp[i] >= a.Parch or cp[j] >= b.Parch):
                continue
            if l == "spouse" and (sp[i] or sp[j]):
                continue
            if (l == "parent" and pa[i] >= 2) or (l == "child" and pa[j] >= 2):
                continue
            if l in ("sibling", "spouse"):
                cs[i] += 1; cs[j] += 1
            else:
                cp[i] += 1; cp[j] += 1
            if l == "spouse":
                sp[i] += 1; sp[j] += 1
            if l == "parent":
                pa[i] += 1
            if l == "child":
                pa[j] += 1
            labels[(min(i, j), max(i, j))] = l if i < j else FLIP[l]
            if ri != rj:
                members[ri] |= members.pop(rj)
                for k in members[ri]:
                    root[k] = ri
            break
    return pd.Series(root), {r: frozenset(m) for r, m in members.items()}, labels


def candidates_v3(p):
    """Like candidates(), but pairs are kept when any v3 label other than none is possible."""
    fam = p[p["Expected"] > 1]
    rows = []
    for (i, a), (j, b) in combinations(fam.iterrows(), 2):
        roles = [l for l in allowed(a, b) if l != "none"]
        if not roles:
            continue
        m = matches(a, b)
        rows.append({"i": i, "j": j, "A": a.PassengerId, "B": b.PassengerId,
                     "Role of B": " or ".join(roles), **m,
                     "Score": sum(m.values()), "Strong": any(m[k] for k in STRONG)})
    return pd.DataFrame(rows)


def _placed(p, mem, lab):
    rel = {i: [0, 0] for i in mem}
    for (a, b), l in lab.items():
        if a in rel and b in rel and l != "none":
            k = 0 if l in ("sibling", "spouse") else 1
            rel[a][k] += 1; rel[b][k] += 1
    return {i: rel[i][0] == p.at[i, "SibSp"] and rel[i][1] == p.at[i, "Parch"] for i in mem}


def check_people(p, households, link_labels, search_up_to=8):
    """Per-person result. Starts from the labels given while joining. Where someone is
    not fully placed and the household has at most search_up_to people, a full search
    over every pair label looks for a labelling that places more people."""
    status, labels = {}, {}
    for r, mem in households.items():
        if len(mem) == 1:
            status[next(iter(mem))] = "No match found"
            continue
        lab = {k: v for k, v in link_labels.items() if k[0] in mem}
        placed = _placed(p, mem, lab)
        if not all(placed.values()) and len(mem) <= search_up_to:
            n, found, ok = best_labelling(p, mem)
            if ok and n > sum(placed.values()):
                lab = {k: v for k, v in found.items() if v != "none"}
                placed = _placed(p, mem, lab)
        labels[r] = lab
        for i in mem:
            status[i] = "Fully placed" if placed[i] else "Partly placed"
    return pd.Series(status), labels
