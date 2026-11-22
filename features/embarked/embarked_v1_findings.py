# Findings text for embarked_v1.ipynb, read by build_embarked_v1.py. Numbers are from the run on complete_data.csv.

FINDINGS = {
"summary": """## Findings

- **On its own, port is strongly related to survival.** 55% of Cherbourg passengers survived, against 39% at Queenstown and 34% at Southampton (p < 0.001).
- **Most of the Cherbourg gap is who boarded there.** Half of Cherbourg's passengers were 1st class, against a fifth at Southampton. Within the same class and sex, Cherbourg's odds of survival are still 1.87 times Southampton's (p = 0.008), but adding age, relatives aboard, fare, title and deck brings that down to 1.61, with an interval that touches 1 (p = 0.06).
- **Queenstown looks better than Southampton only once class is held fixed**, because almost all Queenstown passengers were 3rd class. After every feature, the odds ratio is 1.53 and not significant (p = 0.24).
- **In the full model, port adds nothing detectable.** The likelihood-ratio test for port gives p = 0.13; cross-validated accuracy rises by 0.5 points (p = 0.39). Port does not matter differently by class or by sex (p = 0.67 and 0.27).
- **One pattern worth knowing:** 3rd class women from Southampton survived at 38%, against 65% from Cherbourg and 73% from Queenstown, and the full model overpredicts them by 16 points. That points to something about that group (for example, the large 3rd class families that boarded at Southampton) rather than to the port itself.
- **Conclusion:** port is a weak feature once class, sex, age, family and fare are known. Keep it in a model if it costs nothing, but expect at most a fraction of a point.""",

"raw": """The raw gap is large: Cherbourg's survival rate is 21 points above Southampton's. The intervals for Southampton and Queenstown overlap.""",

"mix": """Cherbourg's passengers were richer: 51% 1st class, median fare 29.70 GBP against 13.00 at Southampton. Queenstown's were almost all 3rd class (94%) and paid least. Southampton carried the most 3rd class men (41% of its passengers), the group that survived worst. These differences alone would make Cherbourg look better.""",

"strat": """Within class and sex, much of the gap closes, but not all. 1st class women survived at 96 to 98% wherever they boarded. The clearest remaining differences are in 3rd class: women from Southampton survived at 38% against 65% from Cherbourg and 73% from Queenstown, and men from Cherbourg at 23% against 13% from Southampton. The pooled test still finds Cherbourg better than Southampton (odds ratio 1.87, p = 0.008); Queenstown against Southampton is borderline (1.92, p = 0.06).""",

"lr": """Cherbourg's odds ratio falls steadily as features are added, from 2.44 to about 1.5 to 1.6, and stops being significant once fare enters: part of what looked like a port effect was the higher fares paid at Cherbourg. Queenstown's odds ratio jumps once class is held fixed (to 2.33), because Queenstown's passengers were almost all 3rd class, then fades below significance as sex, family and fare enter.""",

"lrt": """None of the tests is significant. Adding port to the full model improves its fit no more than chance would (p = 0.13), and allowing port to work differently by class or by sex adds nothing (p = 0.67 and 0.27).""",

"cv": """Port moves held-out accuracy by about half a point, well within the noise (p of 0.28 to 0.39). Interactions with class or sex do not help either.""",

"res": """A model without port slightly underpredicts Cherbourg (by 4 points over passengers, 9 points over parties) and overpredicts Southampton by about 2 points. Neither test is significant at 0.05: p = 0.14 over passengers, p = 0.06 over parties. The largest leftover gap is not a port as a whole but one group within it: 3rd class women from Southampton, overpredicted by 16 points across 88 passengers.""",
}
