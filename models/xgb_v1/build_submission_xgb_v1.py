"""Train the best model from models/xgb_v1/xgb_v1.ipynb on all 891 train passengers and predict the 418 test passengers.

Model: random setting 305, feature set "+ all three survival". Writes models/xgb_v1/submission_xgb_v1.csv. Features are built exactly as in build_xgb_v1.py. Reads data/complete_data.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
c = pd.read_csv(ROOT / "data" / "complete_data.csv")

t = c["Name"].str.extract(r",\s*([^.]+)\.", expand=False).str.strip().replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
c["Title"] = t.where(t.isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")
c["SexF"] = (c["Sex"] == "female").astype(int)
c["HasCabin"] = c["Cabin"].notna().astype(int)
for col in ["Title", "Deck", "Embarked"]:
    c[col] = pd.Categorical(c[col], categories=sorted(c[col].dropna().unique()))

def parse(s):
    out = []
    if isinstance(s, str):
        for part in s.split(";"):
            role, ids = part.strip().split(" ", 1)
            out += [(role, int(i)) for i in ids.split(",")]
    return out
rel = dict(zip(c["PassengerId"], c["Relatives"].map(parse)))
sex = dict(zip(c["PassengerId"], c["Sex"]))
fam = dict(zip(c["PassengerId"], c["FamilyId"]))
tick = dict(zip(c["PassengerId"], c["Ticket"]))
on_ticket = c.groupby("Ticket")["PassengerId"].apply(list).to_dict()
is_tr = c["Dataset"] == "train"
surv = dict(zip(c.loc[is_tr, "PassengerId"], c.loc[is_tr, "Survived"]))

def share(ids, pid):
    v = [surv[o] for o in ids if o in surv and o != pid]
    return float(np.mean(v)) if v else np.nan
c["fem_rel_survival"] = c["PassengerId"].map(lambda p: share(
    [o for role, o in rel[p] if sex[o] == "female" and (role == "parent" or (sex[p] == "female" and role in ("sibling", "child")))], p))
c["family_survival"] = c["PassengerId"].map(lambda p: share([o for _, o in rel[p]], p))
def ticket_nonfam(p):
    rels, f = {o for _, o in rel[p]}, fam[p]
    return share([o for o in on_ticket[tick[p]] if o not in rels and not (pd.notna(f) and fam[o] == f)], p)
c["ticket_survival"] = c["PassengerId"].map(ticket_nonfam)

# Checks against the feature spec's worked examples
ex = c.set_index("PassengerId")
assert ex.loc[945, "fem_rel_survival"] == 1.0 and ex.loc[945, "family_survival"] == 0.5
assert np.isnan(ex.loc[439, "fem_rel_survival"]) and abs(ex.loc[439, "family_survival"] - 2 / 3) < 1e-9
assert ex.loc[1032, "fem_rel_survival"] == 0.0 and ex.loc[1032, "family_survival"] == 0.0

FEATURES = ["Pclass", "Title", "SexF", "Age", "HasCabin", "Deck", "SibSp", "Parch",
            "fem_rel_survival", "family_survival", "ticket_survival"]
PARAMS = dict(max_depth=6, learning_rate=0.0431430, n_estimators=800, min_child_weight=1.490898, gamma=1.92653,
              subsample=0.939319, colsample_bytree=0.437225, reg_lambda=0.111106, reg_alpha=0.0)

tr, te = c[is_tr], c[~is_tr]
m = xgb.XGBClassifier(**PARAMS, objective="binary:logistic", tree_method="hist", enable_categorical=True,
                      n_jobs=1, random_state=0)
m.fit(tr[FEATURES], tr["Survived"].astype(int))
p = m.predict_proba(te[FEATURES])[:, 1]

sub = pd.DataFrame({"PassengerId": te["PassengerId"].astype(int).values, "Survived": (p >= 0.5).astype(int)})
assert len(sub) == 418 and sub["PassengerId"].is_unique
out = Path(__file__).resolve().parent / "submission_xgb_v1.csv"
sub.to_csv(out, index=False)
train_acc = ((m.predict_proba(tr[FEATURES])[:, 1] >= 0.5) == tr["Survived"]).mean()
print(f"written {out}: {len(sub)} rows, {sub['Survived'].mean():.1%} predicted to survive, "
      f"train accuracy {train_acc:.1%} (fitted rows, not a holdout score)")
