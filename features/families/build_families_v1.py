"""Build features/families/families_v1.ipynb: family groups by surname, ticket and cabin, code cells hidden.

Superseded by build_families_v2.py. Running this overwrites data/*_families.* with the v1 result."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
HIDE = {"jupyter": {"source_hidden": True}}

intro = """# Families

A passenger with SibSp + Parch = 0 travelled with no family and has no family ID. A passenger with SibSp + Parch = x belongs to a family of x + 1 people, so x other passengers should share their family.

train.csv and test.csv are grouped together, because Kaggle split the passenger list at random and many families have members in both files. Only passengers with SibSp + Parch above 0 are grouped.

**Check.** A group is correct when its size equals SibSp + Parch + 1 for every member of it. The check tests group sizes, not who is in the group, so a correct size is strong evidence but not proof."""

method = """## Method

**Run 1: surname only.** Every passenger is grouped with every other passenger who has the same surname (the text before the comma in `Name`).

**Run 2: surname, ticket and cabin.** Passengers are joined in this order, strongest evidence first:

1. Same surname and same ticket
2. Same ticket
3. Same cabin. A passenger with several cabins, such as `C23 C25 C27`, is joined to anyone sharing any one of them
4. Same surname

A join is made only when the combined group is no bigger than SibSp + Parch + 1 for every person in it. This stops a common surname such as Andersson from swallowing unrelated families, and stops a shared ticket from pulling in more people than the family says it has.

After that, any groups still failing the check that share a ticket or a cabin are joined regardless of size and marked **Linked, sizes disagree**."""

load = r'''import pandas as pd
from IPython.display import display, Markdown

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
people = pd.concat([train, test], ignore_index=True).sort_values("PassengerId", ignore_index=True)
people["Surname"] = people["Name"].str.split(",").str[0].str.strip()
people["Expected"] = people["SibSp"] + people["Parch"] + 1
fam = people[people["Expected"] > 1].copy()

cabins = (
    fam["Cabin"].dropna().str.split().explode().rename("CabinNo").reset_index()
    .rename(columns={"index": "row"})
)

def groups(keys):
    """Lists of row labels that share the given key(s)."""
    if keys == "Cabin":
        return [list(g["row"]) for _, g in cabins.groupby("CabinNo") if len(g) > 1]
    return [list(ix) for ix in fam.groupby(keys).groups.values() if len(ix) > 1]

class Families:
    def __init__(self, rows):
        self.parent = {r: r for r in rows}
        self.size = {r: 1 for r in rows}
        self.cap = {r: fam.at[r, "Expected"] for r in rows}
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def join(self, a, b, capped=True):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if capped and self.size[ra] + self.size[rb] > min(self.cap[ra], self.cap[rb]):
            return
        self.parent[ra] = rb
        self.size[rb] += self.size[ra]
        self.cap[rb] = min(self.cap[ra], self.cap[rb])
    def labels(self):
        return pd.Series({r: self.find(r) for r in self.parent})

def check(label):
    g = fam.assign(G=label).groupby("G")["Expected"].agg(["size", "min", "max"])
    return ((g["min"] == g["max"]) & (g["size"] == g["min"])).rename("Correct")

# Run 1: surname only
run1 = Families(fam.index)
for ix in groups("Surname"):
    for r in ix[1:]:
        run1.join(r, ix[0], capped=False)
fam["Run1"] = run1.labels()

# Run 2: surname + ticket, ticket, cabin, surname, each capped at the family size
run2 = Families(fam.index)
for keys in [["Surname", "Ticket"], "Ticket", "Cabin", "Surname"]:
    for ix in groups(keys):
        for r in ix[1:]:
            run2.join(r, ix[0])
lab = run2.labels()
ok = check(lab)
correct_rows = set(lab[lab.map(ok)].index)

# Second pass: join still-failing groups that share a ticket or cabin, ignoring size
for keys in ["Ticket", "Cabin"]:
    for ix in groups(keys):
        ix = [r for r in ix if r not in correct_rows]
        for r in ix[1:]:
            run2.join(r, ix[0], capped=False)
fam["Run2"] = run2.labels()

sizes = fam.groupby("Run2")["PassengerId"].transform("size")
fam["FamilyCheck"] = "Correct"
fam.loc[~fam.index.isin(correct_rows) & (sizes > 1), "FamilyCheck"] = "Linked, sizes disagree"
fam.loc[~fam.index.isin(correct_rows) & (sizes == 1), "FamilyCheck"] = "No relative found"

# FamilyId from 1, in order of each family's first PassengerId
first = fam.groupby("Run2")["PassengerId"].min().sort_values()
fam["FamilyId"] = fam["Run2"].map(pd.Series(range(1, len(first) + 1), index=first.index))
people = people.join(fam[["FamilyId", "FamilyCheck"]])
people["FamilyId"] = people["FamilyId"].astype("Int64")

def with_family(src):
    cols = list(src.columns)
    at = cols.index("Parch") + 1
    cols[at:at] = ["FamilyId", "FamilyCheck"]
    return src.merge(people[["PassengerId", "FamilyId", "FamilyCheck"]], on="PassengerId")[cols]

train_out, test_out = with_family(train), with_family(test)
train_out.to_csv("data/train_families.csv", index=False)
test_out.to_csv("data/test_families.csv", index=False)

GREEN = "#d9f2d0"
with pd.ExcelWriter("data/families.xlsx", engine="openpyxl") as xl:
    for sheet, out in [("train", train_out), ("test", test_out)]:
        out.style.set_properties(subset=["FamilyId", "FamilyCheck"], **{"background-color": GREEN}) \
            .to_excel(xl, sheet_name=sheet, index=False)
        xl.sheets[sheet].freeze_panes = "A2"

n = len(fam)
r1 = int(fam["Run1"].map(check(fam["Run1"])).sum())
r2 = len(correct_rows)
display(Markdown(
    f"{len(people):,} passengers. {len(people) - n:,} travelled alone and have no FamilyId. {n:,} have family aboard.\n\n"
    f"**Run 1 (surname): {r1} of {n} correct ({100 * r1 / n:.0f}%). "
    f"Run 2 (surname, ticket and cabin): {r2} of {n} correct ({100 * r2 / n:.0f}%).**"
))'''

results_md = "## Results"

results = r'''def tally(label, verdict):
    t = pd.DataFrame({"G": label, "V": verdict}).groupby("V").agg(
        Groups=("G", "nunique"), Passengers=("G", "size"))
    return t

r1_ok = fam["Run1"].map(check(fam["Run1"]))
r1_size = fam.groupby("Run1")["PassengerId"].transform("size")
r1_v = pd.Series("Correct", index=fam.index)
r1_v[~r1_ok & (r1_size > 1)] = "Linked, sizes disagree"
r1_v[~r1_ok & (r1_size == 1)] = "No relative found"

order = ["Correct", "Linked, sizes disagree", "No relative found"]
t = pd.concat({"Run 1: surname": tally(fam["Run1"], r1_v), "Run 2: surname, ticket, cabin": tally(fam["Run2"], fam["FamilyCheck"])}, axis=1)
t = t.reindex(order).fillna(0).astype(int)
t.loc["Total"] = t.sum()
t.index.name = None
t'''

found_md = """## Relatives found by ticket or cabin

Families whose members have more than one surname. Surname alone could not have put these people together. Every one of them passes the check."""

found = r'''mixed = (
    fam[fam["FamilyCheck"] == "Correct"].groupby("FamilyId")
    .filter(lambda g: g["Surname"].nunique() > 1)
    .sort_values(["FamilyId", "PassengerId"])
)
display(Markdown(f"{mixed['FamilyId'].nunique()} families, {len(mixed)} passengers."))
mixed[["FamilyId", "PassengerId", "Name", "SibSp", "Parch", "Ticket", "Cabin"]] \
    .style.format(na_rep="").hide(axis="index")'''

left_md = """## Still not resolved

**Linked, sizes disagree.** These passengers share a ticket, a cabin or a surname, but their SibSp + Parch counts cannot all be true for one closed group. Most are households where the counts are not the same for everyone: a husband counts his wife and his own siblings, while his wife counts only him (Vander Planke), or a mother counts her children and her own parents and siblings travelling separately (Richards, née Hocking). These are likely real families that the "x + 1 people" rule cannot describe.

**No relative found.** No other passenger with family aboard shares their surname, ticket or cabin within the size limit. Several are married women whose relatives travel under her maiden name, which appears in brackets in `Name`: Wilkes (Ellen Needs) is sister to Hocking (Eliza Needs), and Renouf (Lillian Jefferys) is sister to the Jefferys brothers. Matching the bracketed maiden name to a surname would recover some of these."""

left = r'''rest = fam[fam["FamilyCheck"] != "Correct"].sort_values(["FamilyCheck", "FamilyId", "PassengerId"])
rest[["FamilyCheck", "FamilyId", "PassengerId", "Name", "SibSp", "Parch", "Expected", "Ticket", "Cabin"]] \
    .rename(columns={"Expected": "Family size"}).style.format(na_rep="").hide(axis="index")'''

preview = r'''display(Markdown("## train_families.csv, first 25 rows\n\nFamilyId and FamilyCheck are the light green columns."))
train_out.head(25).style.set_properties(subset=["FamilyId", "FamilyCheck"], **{"background-color": GREEN}) \
    .format(na_rep="", precision=2).hide(axis="index")'''

files_md = """## Files written

- `data/train_families.csv` and `data/test_families.csv`: the original columns with `FamilyId` and `FamilyCheck` added straight after `Parch`, both blank for passengers with no family. `FamilyId` comes from run 2
- `data/families.xlsx`: the same data on a train sheet and a test sheet, with the two new columns shaded light green. A CSV is plain text and cannot hold a colour, so the shading lives in the xlsx

The original train.csv and test.csv are unchanged."""

nb = nbformat.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
nb.cells = [
    nbformat.v4.new_markdown_cell(intro),
    nbformat.v4.new_markdown_cell(method),
    nbformat.v4.new_code_cell(load, metadata=HIDE),
    nbformat.v4.new_markdown_cell(results_md),
    nbformat.v4.new_code_cell(results, metadata=HIDE),
    nbformat.v4.new_markdown_cell(found_md),
    nbformat.v4.new_code_cell(found, metadata=HIDE),
    nbformat.v4.new_markdown_cell(left_md),
    nbformat.v4.new_code_cell(left, metadata=HIDE),
    nbformat.v4.new_code_cell(preview, metadata=HIDE),
    nbformat.v4.new_markdown_cell(files_md),
]
NotebookClient(nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
nbformat.write(nb, ROOT / "features" / "families" / "families_v1.ipynb")
print("written v1")
