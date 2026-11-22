"""Add a Deck column, straight after Cabin, to data/complete_data.csv, train_families.csv and test_families.csv.

Deck is the first letter of the passenger's own Cabin. Where Cabin is blank, it is taken from other passengers
on the exact same ticket who have a cabin recorded (most common deck among them). Otherwise blank.

Exact ticket-mates agree on deck for 194 of 198 passengers scored leave-one-out. Neighbouring ticket numbers
do not: the nearest other 1st class ticket within 2 numbers shares the deck only 32% of the time, so no wider
window is used.

Re-run after build_families_v3.py, which rewrites these files without Deck.
"""
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parents[2] / "data"
c = pd.read_csv(D / "complete_data.csv")

own = c["Cabin"].str[0]
mates = (c.assign(D=own).dropna(subset=["D"]).groupby("Ticket")["D"]
         .agg(lambda s: s.value_counts().index[0]))
deck = own.fillna(c["Ticket"].map(mates))
by_id = dict(zip(c["PassengerId"], deck))

for name in ["complete_data.csv", "train_families.csv", "test_families.csv"]:
    df = pd.read_csv(D / name)
    for col in ["Survived", "FamilyId"]:
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    df = df.drop(columns=["Deck"], errors="ignore")
    df.insert(df.columns.get_loc("Cabin") + 1, "Deck", df["PassengerId"].map(by_id))
    df.to_csv(D / name, index=False)

print(f"own cabin: {own.notna().sum()}, from ticket-mates: {(deck.notna() & own.isna()).sum()}, "
      f"blank: {deck.isna().sum()} of {len(c)}")
