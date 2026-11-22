"""Build features/families/families_v2.ipynb from features/families/family_logic.py, code cells hidden.

Superseded by build_families_v3.py. Running this overwrites data/*_families.* and data/complete_data.* with the v2 result.
"""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
AREA = ROOT / "features" / "families"
HIDE = {"jupyter": {"source_hidden": True}}

intro = """# Families v2: relationship rules plus feature matching

Every passenger with SibSp + Parch above 0 is placed in a family using the rules below. train.csv and test.csv are used together, 1,309 passengers, because families are split across the two files. The rules live in `family_logic.py` in this folder, so they can be changed and the notebook rebuilt.

## Rules

**Step 1: who can be related, from SibSp and Parch alone.** For a passenger with SibSp = x and Parch = y, in a family where everyone is aboard:

- a **sibling** has SibSp = x and Parch = y
- a **spouse** has SibSp of at least 1 and the same Parch, because they share the same children
- a **parent** has Parch = x + 1 (this passenger and their x siblings), and SibSp of at least 1 when y = 2 (the other parent)
- a **child** is the reverse of a parent

Two passengers who fit none of these cannot be in the same family.

**Step 2: what they share.** Every pair that passes step 1 is scored on six features: surname, ticket, cabin, class, port of embarkation and fare. A surname counts as matching when it equals the other passenger's surname or their maiden name in brackets, for example Hogeboom, Mrs. John C (Anna **Andrews**). The more features match, the stronger the link.

**Step 3: build families, strongest links first.** A pair is only joined when it shares surname, ticket or cabin, because class and port alone match thousands of strangers. A join is allowed only when:

- the family stays no bigger than SibSp + Parch + 1 for every member, and
- every pair inside the family passes step 1

**Result.** A family is **complete** when its size equals SibSp + Parch + 1 for every member."""

load = r'''import pandas as pd
from IPython.display import display, Markdown
import family_logic as fl

train, test, people = fl.load()
fam = people[people["Expected"] > 1]
cand = fl.candidates(people)
label = fl.build(people, cand)
result = fl.check(people, label)

n = len(fam)
counts = result.value_counts()
display(Markdown(
    f"{len(people):,} passengers. {len(people) - n:,} have SibSp + Parch = 0 and no family. "
    f"{n} have family aboard.\n\n"
    f"**{counts.get('Complete family', 0)} of {n} ({100 * counts.get('Complete family', 0) / n:.0f}%) are placed in a complete family.**"
))'''

narrow_md = """## How far each step narrows the search

Of the 134,421 possible pairs among the 519 passengers with family aboard, step 1 keeps the pairs whose SibSp and Parch allow a relationship. Step 2 then keeps the pairs that also share surname, ticket or cabin."""

narrow = r'''total = n * (n - 1) // 2
per_person = pd.concat([cand["i"], cand["j"]]).value_counts().reindex(fam.index, fill_value=0)
strong = cand[cand["Strong"]]
per_strong = pd.concat([strong["i"], strong["j"]]).value_counts().reindex(fam.index, fill_value=0)
pd.DataFrame({
    "Pairs": [total, len(cand), len(strong)],
    "Median candidates per passenger": [n - 1, int(per_person.median()), int(per_strong.median())],
    "Passengers with no candidate": [0, int((per_person == 0).sum()), int((per_strong == 0).sum())],
}, index=["All pairs", "Pass step 1 (SibSp and Parch)", "Also share surname, ticket or cabin"])'''

feat_md = """## Which features mark a family

For each feature, the share of pairs that match it: pairs inside the same complete family, against pairs that pass step 1 but end up in different families. A feature is useful when the first number is high and the second is low."""

feat = r'''pair_fam = cand.assign(Gi=cand["i"].map(label), Gj=cand["j"].map(label))
complete = set(label[result == "Complete family"].index)
same = pair_fam[(pair_fam["Gi"] == pair_fam["Gj"]) & pair_fam["i"].isin(complete)]
diff = pair_fam[pair_fam["Gi"] != pair_fam["Gj"]]
tab = pd.DataFrame({
    "Same family": (100 * same[fl.FEATURES].mean()).round(0).astype(int).astype(str) + "%",
    "Different families": (100 * diff[fl.FEATURES].mean()).round(1).astype(str) + "%",
})
tab.index.name = "Feature matches"
display(Markdown(f"{len(same):,} pairs in the same family, {len(diff):,} in different families."))
tab'''

score_md = """Number of features matched, out of six, for pairs in the same family. No family pair matches on fewer than three, and 480 of 508 match on five or six."""

score = r'''sc = pd.DataFrame({
    "Same family": same["Score"].value_counts(),
    "Different families": diff["Score"].value_counts(),
}).reindex(range(7), fill_value=0).fillna(0).astype(int)
sc.index.name = "Features matched"
sc.T'''

res_md = "## Results"

res = r'''v1 = pd.DataFrame({
    "Surname only": [401, 89, 29],
    "Surname, ticket, cabin": [445, 54, 20],
}, index=["Complete family", "Incomplete family", "No match found"])
v1["Rules plus features (this notebook)"] = counts.reindex(v1.index).fillna(0).astype(int)
v1.loc["Total"] = v1.sum()
v1'''

res_note = """All 445 passengers from the surname, ticket and cabin run land in the same families here, and every pair inside them passes step 1. The 3 gained are the Davies brothers (Alfred, John and Joseph), who share a surname and near tickets but no single ticket."""

left_md = """## What the rules cannot place

The rules assume a closed family: everyone in it counts the same set of relatives. Most of the passengers left over belong to households where that is not true, so no single family can have the size every member asks for:

- **A married sibling.** Julius Vander Planke has SibSp 3: his wife plus his brother Leo and sister Augusta. His wife has SibSp 1, only him. Leo and Augusta have SibSp 2. The four are one household, but the wife wants a family of 2, Julius a family of 4 and his siblings a family of 3.
- **Three generations.** Mrs Sidney Richards, born Emily Hocking, has Parch 3: her two sons plus her mother, Mrs Elizabeth Hocking. Her sons count only her. The Richards and Hocking passengers are one household under two surnames and four tickets.
- **Relatives who share nothing.** Ellen Wilkes (born Needs) is sister to Eliza Hocking (born Needs), but neither surname, ticket nor cabin matches, and the maiden name of one does not match the married surname of the other.

Placing these needs a different output: a list of each passenger's relatives, or a household ID that is allowed to be larger than some members' SibSp + Parch + 1."""

left = r'''rest = people.loc[result.index].assign(Group=label, Result=result)
rest = rest[rest["Result"] != "Complete family"]
first = rest.groupby("Group")["PassengerId"].transform("min")
rest.assign(First=first).sort_values(["Result", "First", "PassengerId"])[
    ["Result", "PassengerId", "Name", "SibSp", "Parch", "Ticket", "Cabin", "Pclass", "Embarked"]
].style.format(na_rep="").hide(axis="index")'''

files = r'''order = fam.assign(G=label).groupby("G")["PassengerId"].min().sort_values()
fid = label.map(pd.Series(range(1, len(order) + 1), index=order.index))
people["FamilyId"] = fid.astype("Int64")
people["FamilyCheck"] = result

def with_family(src):
    cols = list(src.columns)
    at = cols.index("Parch") + 1
    cols[at:at] = ["FamilyId", "FamilyCheck"]
    return src.merge(people[["PassengerId", "FamilyId", "FamilyCheck"]], on="PassengerId")[cols]

d = fl.data_dir()
train_out, test_out = with_family(train), with_family(test)
train_out.to_csv(d / "train_families.csv", index=False)
test_out.to_csv(d / "test_families.csv", index=False)
GREEN = "#d9f2d0"
with pd.ExcelWriter(d / "families.xlsx", engine="openpyxl") as xl:
    for sheet, out in [("train", train_out), ("test", test_out)]:
        out.style.set_properties(subset=["FamilyId", "FamilyCheck"], **{"background-color": GREEN}) \
            .to_excel(xl, sheet_name=sheet, index=False)
        xl.sheets[sheet].freeze_panes = "A2"

# Complete data: train and test in one file, PassengerId order, Dataset first
complete = pd.concat([train_out.assign(Dataset="train"), test_out.assign(Dataset="test")], ignore_index=True)
complete = complete[["Dataset"] + list(train_out.columns)].sort_values("PassengerId", ignore_index=True)
complete["Survived"] = complete["Survived"].astype("Int64")
complete.to_csv(d / "complete_data.csv", index=False)
TEXT = {"train": "#1f4fd1", "test": "#d11f1f"}
with pd.ExcelWriter(d / "complete_data.xlsx", engine="openpyxl") as xl:
    complete.style \
        .apply(lambda r: [f"color: {TEXT[r['Dataset']]}"] * len(r), axis=1) \
        .set_properties(subset=["FamilyId", "FamilyCheck"], **{"background-color": GREEN}) \
        .to_excel(xl, sheet_name="complete", index=False)
    xl.sheets["complete"].freeze_panes = "B2"

pairs = cand[cand["Strong"]].drop(columns=["i", "j", "Strong"]).rename(columns={"A": "PassengerId A", "B": "PassengerId B"})
pairs = pairs.sort_values(["Score", "PassengerId A"], ascending=[False, True])
pairs.to_csv("family_candidates.csv", index=False)

display(Markdown(
    "## Files written\n\n"
    "- `data/train_families.csv` and `data/test_families.csv`: `FamilyId` and `FamilyCheck` straight after `Parch`, "
    "blank for passengers with no family. These now come from this notebook\n"
    "- `data/families.xlsx`: the same, with the two columns shaded light green\n"
    f"- `data/complete_data.csv`: train and test in one file, {len(complete):,} passengers in PassengerId order. "
    "The first column, `Dataset`, is train or test. `Survived` is blank for test passengers\n"
    "- `data/complete_data.xlsx`: the same, with train passengers in blue text, test passengers in red text "
    "and the family columns shaded light green\n"
    f"- `features/families/family_candidates.csv`: the {len(pairs)} candidate pairs that pass step 1 and share surname, "
    "ticket or cabin, with the role, each feature match and the score. Pairs that match only on class, port or fare are left out"
))'''

nb = nbformat.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
md, code = nbformat.v4.new_markdown_cell, lambda s: nbformat.v4.new_code_cell(s, metadata=HIDE)
nb.cells = [md(intro), code(load), md(narrow_md), code(narrow), md(feat_md), code(feat), md(score_md), code(score),
            md(res_md), code(res), md(res_note), md(left_md), code(left), code(files)]
NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(AREA)}}).execute()
nbformat.write(nb, AREA / "families_v2.ipynb")
print("written", AREA / "families_v2.ipynb")
