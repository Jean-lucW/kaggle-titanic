"""Shared feature build for the XGBoost models: same definitions as build_xgb_v1.py.

build() returns data/complete_data.csv with Title, SexF, HasCabin, the three survival features (train outcomes only,
never the passenger's own) and Group, the linked ticket/family/relative group used to keep folds leak-free.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def build():
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

    parent = {p: p for p in c["PassengerId"]}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def link(ids):
        ids = list(ids)
        for i in ids[1:]:
            parent[find(i)] = find(ids[0])
    for _, ids in c.groupby("Ticket")["PassengerId"]:
        link(ids)
    for _, ids in c.dropna(subset=["FamilyId"]).groupby("FamilyId")["PassengerId"]:
        link(ids)
    for p, rs in rel.items():
        link([p] + [o for _, o in rs])
    c["Group"] = c["PassengerId"].map(find)

    ex = c.set_index("PassengerId")
    assert ex.loc[945, "fem_rel_survival"] == 1.0 and ex.loc[945, "family_survival"] == 0.5
    assert np.isnan(ex.loc[439, "fem_rel_survival"]) and abs(ex.loc[439, "family_survival"] - 2 / 3) < 1e-9
    assert ex.loc[1032, "fem_rel_survival"] == 0.0 and ex.loc[1032, "family_survival"] == 0.0
    return c
