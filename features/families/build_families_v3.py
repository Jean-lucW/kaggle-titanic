"""Build features/families/families_v3.ipynb (households and the per-person check) and write the data files.

The data files (data/*_families.*, data/complete_data.*) come from this script.
Running build_families_v1.py or build_families_v2.py overwrites them with the older results.
"""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
AREA = ROOT / "features" / "families"
HIDE = {"jupyter": {"source_hidden": True}}

intro = """# Families v3: households and the per-person check

v2 required every family to be closed: everyone in it counts the same relatives, so the family size equals SibSp + Parch + 1 for every member. Real households break that. Julius Vander Planke counts his wife and two siblings, while his wife counts only him. v3 drops the closed-family rule and checks each person instead.

train.csv and test.csv are used together, 1,309 passengers. The rules live in `family_logic.py` in this folder.

## Rules

**Who can be linked.** Passengers with SibSp + Parch above 0 are linked when they share any of:

- **surname**: the same surname, one person's maiden name (in brackets after Mrs) equal to the other's surname, the same maiden name on both, or one part of a hyphenated surname (Kink and Kink-Heilmann)
- **ticket**
- **cabin**: any one cabin in common

**Labelling each pair.** Inside a household every pair is labelled sibling, spouse, parent, child or none. None means the two are in the same household but not directly related, such as a man and his brother's wife. A label is only allowed when:

- **sibling**: both have SibSp of at least 1
- **spouse**: a man (not a Master) and a Mrs with the same surname, both with SibSp of at least 1, at most one spouse each. When her name gives her husband's first name ("Mrs. Julius"), his first name must match
- **parent and child**: both have Parch of at least 1, the parent is not a Master, at most two parents each, and the parent is at least 14 years older when both ages are known
- nobody gets more sibling or spouse labels than their SibSp, or more parent or child labels than their Parch

Links are labelled strongest first, ranked by how many of surname, ticket, cabin, class, port and fare the two share. Where a pair can take more than one label, the label that fits the closed-family rule from v2 is tried first.

**The per-person check.** A passenger is **fully placed** when their household holds exactly SibSp people labelled sibling or spouse and exactly Parch people labelled parent or child. Where someone in a household of 8 or fewer is not fully placed, every labelling of the household is searched for one that places more people."""

load = r'''import pandas as pd
from IPython.display import display, Markdown
import family_logic as fl

train, test, people = fl.load()
people = fl.add_person_fields(people)
cand = fl.candidates_v3(people)
household, households, link_labels = fl.build_households(people, cand)
status, labels = fl.check_people(people, households, link_labels)

fam = people.loc[status.index]
n = len(fam)
counts = status.value_counts()
full = counts.get("Fully placed", 0)

# Each person's relatives, read from the labels
pid = people["PassengerId"]
rel = {i: [] for i in fam.index}
for lab in labels.values():
    for (a, b), l in lab.items():
        rel[a].append((l, pid[b]))
        rel[b].append((fl.FLIP[l], pid[a]))
def describe(items):
    out = []
    for role in ["spouse", "sibling", "parent", "child"]:
        ids = sorted(int(x) for r, x in items if r == role)
        if ids:
            out.append(f"{role} {', '.join(map(str, ids))}")
    return "; ".join(out)
people["Relatives"] = pd.Series({i: describe(v) for i, v in rel.items()})

display(Markdown(
    f"{len(people):,} passengers. {len(people) - n:,} have SibSp + Parch = 0 and no family. {n} have family aboard.\n\n"
    f"**{full} of {n} ({100 * full / n:.0f}%) are fully placed.**"
))'''

res = r'''tab = pd.DataFrame({
    "v1 surname only": [401, 89, 29],
    "v1 surname, ticket, cabin": [445, 54, 20],
    "v2 rules plus features": [448, 51, 20],
    "v3 per-person check": [counts.get(k, 0) for k in ["Fully placed", "Partly placed", "No match found"]],
}, index=["Fully placed (v1, v2: complete family)", "Partly placed (v1, v2: incomplete family)", "No match found"])
tab.loc["Total"] = tab.sum()
tab'''

ex_md = """## Households v2 could not place

Each passenger's relatives as labelled, by PassengerId."""

ex = r'''def show(names, note):
    rows = people[people["Surname"].isin(names)]
    hh = set(household[rows.index[rows.index.isin(household.index)]])
    g = people.loc[[i for i in fam.index if household[i] in hh]].sort_values("PassengerId")
    display(Markdown(note))
    display(g.assign(Result=status)[["PassengerId", "Name", "Age", "SibSp", "Parch", "Ticket", "Relatives", "Result"]]
            .style.format(na_rep="", precision=0).hide(axis="index"))

show(["Vander Planke"], "**Vander Planke.** A married brother with his wife, his brother and his sister. Julius counts 3 (wife, brother, sister), his wife counts 1 and his siblings 2 each. Everyone's count is met in one household of 4.")
show(["Davidson"], "**Davidson and Hays.** Mrs Thornton Davidson, born Orian Hays, travels with her husband and her parents. Her maiden name links her to the Hays couple. v2 left Mr Davidson out because his Parch of 0 differs from hers.")
show(["Richards"], "**Hocking, Richards and Wilkes.** Three generations under three surnames and four tickets. Mrs Eliza Hocking is mother to Emily Richards (born Hocking), Richard and Ellen. Her sister Mrs Ellen Wilkes shares her maiden name, Needs.")
show(["Strom"], "**Strom and Persson.** Mrs Wilhelm Strom, born Persson, travels with her daughter on one ticket and her brother Ernst Persson on another. He cannot be her husband, because a wife carries her husband's surname.")'''

left_md = """## Not fully placed

**Partly placed** passengers are in a household, but their household cannot meet their SibSp and Parch. **No match found** passengers share no surname, maiden name, ticket or cabin with anyone who has room for them. The main causes:

- **Relatives not in the data.** Kaggle's 1,309 passengers are most but not all of those aboard, so a sibling or child counted in SibSp or Parch may be missing. Hamalainen, Lahtinen and Beckwith each count a child who is not on the list, unless that child is under another name.
- **Counts that contradict each other.** Rossmore Abbott counts one sibling, but his brother Eugene counts none. The Ford children count two siblings each although three are aboard. Erna and Ida Andersson copy the 347082 children's counts although that family of 7 is complete.
- **Links the rules do not see.** Helen Newsom is Mrs Beckwith's daughter. Her middle name, Monypeny, is her mother's maiden name, and their tickets are consecutive (11751 and 11752), but neither counts as a match."""

left = r'''rest = fam.assign(Result=status, Household=household, Relatives=people["Relatives"])
rest = rest[rest["Result"] != "Fully placed"]
rest = rest.assign(First=rest.groupby("Household")["PassengerId"].transform("min")).sort_values(["Result", "First", "PassengerId"])
rest[["Result", "PassengerId", "Name", "Age", "SibSp", "Parch", "Ticket", "Cabin", "Relatives"]] \
    .style.format(na_rep="", precision=0).hide(axis="index")'''

files = r'''order = fam.assign(H=household).groupby("H")["PassengerId"].min().sort_values()
people["FamilyId"] = household.map(pd.Series(range(1, len(order) + 1), index=order.index)).astype("Int64")
people["FamilyCheck"] = status

NEW = ["FamilyId", "FamilyCheck", "Relatives"]
def with_family(src):
    cols = list(src.columns)
    at = cols.index("Parch") + 1
    cols[at:at] = NEW
    return src.merge(people[["PassengerId"] + NEW], on="PassengerId")[cols]

d = fl.data_dir()
train_out, test_out = with_family(train), with_family(test)
train_out.to_csv(d / "train_families.csv", index=False)
test_out.to_csv(d / "test_families.csv", index=False)
GREEN = "#d9f2d0"
with pd.ExcelWriter(d / "families.xlsx", engine="openpyxl") as xl:
    for sheet, out in [("train", train_out), ("test", test_out)]:
        out.style.set_properties(subset=NEW, **{"background-color": GREEN}).to_excel(xl, sheet_name=sheet, index=False)
        xl.sheets[sheet].freeze_panes = "A2"

complete = pd.concat([train_out.assign(Dataset="train"), test_out.assign(Dataset="test")], ignore_index=True)
complete = complete[["Dataset"] + list(train_out.columns)].sort_values("PassengerId", ignore_index=True)
complete["Survived"] = complete["Survived"].astype("Int64")
complete.to_csv(d / "complete_data.csv", index=False)
TEXT = {"train": "#1f4fd1", "test": "#d11f1f"}
with pd.ExcelWriter(d / "complete_data.xlsx", engine="openpyxl") as xl:
    complete.style.apply(lambda r: [f"color: {TEXT[r['Dataset']]}"] * len(r), axis=1) \
        .set_properties(subset=NEW, **{"background-color": GREEN}) \
        .to_excel(xl, sheet_name="complete", index=False)
    xl.sheets["complete"].freeze_panes = "B2"

display(Markdown(
    "## Files written\n\n"
    "Three columns sit straight after `Parch`, blank for passengers with no family:\n\n"
    "- `FamilyId`: the household, numbered from 1 in PassengerId order\n"
    "- `FamilyCheck`: Fully placed, Partly placed or No match found\n"
    "- `Relatives`: each relative by role and PassengerId, for example `spouse 19; sibling 39, 334`\n\n"
    "Files:\n\n"
    "- `data/complete_data.csv`: train and test together, PassengerId 1 to 1,309, `Dataset` first\n"
    "- `data/complete_data.xlsx`: the same, train in blue text, test in red text, family columns shaded light green\n"
    "- `data/train_families.csv`, `data/test_families.csv` and `data/families.xlsx`: each dataset on its own"
))'''

nb = nbformat.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
md, code = nbformat.v4.new_markdown_cell, lambda s: nbformat.v4.new_code_cell(s, metadata=HIDE)
nb.cells = [md(intro), code(load), md("## Results"), code(res), md(ex_md), code(ex), md(left_md), code(left), code(files)]
NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(AREA)}}).execute()
nbformat.write(nb, AREA / "families_v3.ipynb")
print("written", AREA / "families_v3.ipynb")
