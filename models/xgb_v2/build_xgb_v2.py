"""Smaller, less overfit XGBoost models: no Deck or HasCabin, trees 1 to 3 levels deep, modest tree counts.

Each candidate is scored two ways:
- the same 3 non-overlapping 25% holdouts as xgb_v1 (StratifiedGroupKFold, 4 parts, seed 0, parts 1 to 3);
- a steadier check: all 4 parts, repeated over 5 shuffles (seeds 0 to 4), 20 holdouts in all.
The pick is the candidate with the lowest average log loss on the 20 holdouts. It is retrained on all 891 train
passengers and written to models/xgb_v2/submission_xgb_v2.csv. All scores go to models/xgb_v2/xgb_v2_results.csv.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from titanic_features import ROOT, build

c = build()
tr = c[c["Dataset"] == "train"].reset_index(drop=True)
te = c[c["Dataset"] == "test"].reset_index(drop=True)
y = tr["Survived"].astype(int).values

BASE = ["Pclass", "Title", "SexF", "Age", "SibSp", "Parch", "fem_rel_survival", "family_survival"]
SETS = {"no deck/cabin": BASE + ["ticket_survival"], "no deck/cabin/ticket": BASE}

GRID = dict(max_depth=[1, 2, 3], n_estimators=[50, 100, 200, 300], learning_rate=[0.03, 0.05, 0.1],
            min_child_weight=[1, 3, 5], reg_lambda=[1, 5], gamma=[0, 1])
configs = [dict(zip(GRID, v), subsample=0.8, colsample_bytree=0.8) for v in itertools.product(*GRID.values())]

splits = {}
for seed in range(5):
    parts = list(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed).split(tr, y, tr["Group"]))
    for k, s in enumerate(parts):
        splits[(seed, k)] = s
ORIG = [(0, 0), (0, 1), (0, 2)]  # the three xgb_v1 holdouts


def run(ci, fs):
    hp, cols, out = configs[ci], SETS[fs], []
    for key, (a, b) in splits.items():
        m = xgb.XGBClassifier(**hp, tree_method="hist", enable_categorical=True, n_jobs=1, random_state=0)
        m.fit(tr.loc[a, cols], y[a])
        p = m.predict_proba(tr.loc[b, cols])[:, 1]
        out.append(dict(config=ci, features=fs, seed=key[0], part=key[1] + 1, orig=key in ORIG,
                        acc=accuracy_score(y[b], p >= 0.5), ll=log_loss(y[b], p), **hp))
    return out


rows = Parallel(n_jobs=-1, prefer="threads")(delayed(run)(ci, fs) for ci in range(len(configs)) for fs in SETS)
R = pd.DataFrame([r for rs in rows for r in rs])
R.to_csv(HERE / "xgb_v2_results.csv", index=False)

allh = R.groupby(["config", "features"]).agg(acc20=("acc", "mean"), ll20=("ll", "mean"), acc_sd=("acc", "std"))
orig = R[R["orig"]].groupby(["config", "features"]).agg(acc3=("acc", "mean"), ll3=("ll", "mean"))
S = allh.join(orig).reset_index().sort_values("ll20")
S = S.join(pd.DataFrame(configs), on="config")
pd.set_option("display.width", 200)
print(f"{len(configs)} settings x {len(SETS)} feature sets x {len(splits)} holdouts = {len(R):,} models\n")
print("Top 10 by log loss on 20 holdouts:")
print(S.head(10)[["features", "max_depth", "n_estimators", "learning_rate", "min_child_weight", "reg_lambda", "gamma",
                  "acc20", "ll20", "acc3", "ll3"]].round(4).to_string(index=False))
print("\nBy depth (median over settings, 20 holdouts):")
print(S.groupby(["features", "max_depth"])[["acc20", "ll20"]].median().round(4))

best = S.iloc[0]
hp = configs[int(best["config"])]
cols = SETS[best["features"]]
m = xgb.XGBClassifier(**hp, tree_method="hist", enable_categorical=True, n_jobs=1, random_state=0).fit(tr[cols], y)
pred = (m.predict_proba(te[cols])[:, 1] >= 0.5).astype(int)
sub = pd.DataFrame({"PassengerId": te["PassengerId"].astype(int), "Survived": pred})
assert len(sub) == 418 and sub["PassengerId"].is_unique
out = HERE / "submission_xgb_v2.csv"
sub.to_csv(out, index=False)
v1 = pd.read_csv(ROOT / "models" / "xgb_v1" / "submission_xgb_v1.csv")
men = (te["Sex"] == "male") & (te["Title"] != "Master")
print(f"\nPicked: {best['features']}, {hp}")
print(f"20 holdouts: accuracy {best['acc20']:.1%} (sd {best['acc_sd']:.1%}), log loss {best['ll20']:.3f}. "
      f"3 original holdouts: accuracy {best['acc3']:.1%}, log loss {best['ll3']:.3f}")
print(f"Submission: {pred.mean():.1%} survive; adult men predicted to live {int(pred[men.values].sum())}; "
      f"differs from v1 on {(sub['Survived'].values != v1['Survived'].values).sum()} passengers; written {out}")
