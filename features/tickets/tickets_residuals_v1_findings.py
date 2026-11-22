# Findings text for tickets_residuals_v1.ipynb, read by build_tickets_residuals_v1.py. Numbers are from the run on complete_data.csv.

FINDINGS = {
"summary": """## Findings

- **The exact ticket carries strong unexplained survival.** Passengers on the same ticket have residuals correlated at 0.48: when the model misjudges one, it misjudges the others the same way. This is the party sharing one fate.
- **The ticket number range does not, once parties are allowed for.** Runs, hundreds, windows, bands and prefixes all explain more residual variation than random groups when counted by passenger (p of 0.02 or less). Counted by party, with each family or ticket group once, none does (p from 0.11 to 0.50). The range effects are parties seen through the number.
- **One exception: tickets 2 to 10 numbers apart.** Passengers on different tickets 2 or 3 numbers apart have residuals correlated at 0.15 (p = 0.001), and 4 to 10 apart at 0.07 (p = 0.01). Both hold for pairs in different parties, so these are not linked families. Likely travelling companions, friends or neighbours who bought tickets together but are not recorded as family. Tickets exactly 1 apart show nothing (0.02), so the pattern is not clean, and beyond 10 apart the correlation is zero.
- **Groups worth a look** (adjusted q below 0.10): ticket 1601 (7 men in 3rd class, 71% survived against 14% expected, one party); run 111426 to 111428 (3 1st class men in 3 separate parties, all survived against 24% expected); run 367226 to 367232 (8 Queenstown 3rd class passengers in 6 parties, 75% survived against 34%); run 364846 to 364851 (6 Queenstown 3rd class passengers in 5 parties, none survived against 46%). The two Queenstown runs suggest groups travelling together without family ties.
- **What to build.** A ticket-group survival feature (the survival rate of other train passengers on the same ticket or in the same family) captures most of this. A second feature for passengers within 2 to 10 numbers on a different ticket may add a little, but it rests on a weak and uneven correlation.""",

"model": """The model without ticket information already scores 82.6% held-out accuracy, higher than the 79.6% basic model in `tickets_model_v1.ipynb`, mainly because title separates boys (Master) from men. Ten repeats move each prediction very little, so the residuals are stable.""",

"level": """Exact ticket explains 66% of residual variation within shared tickets against 39% for random groups of the same sizes: a large, real effect. Every coarser grouping is also significant over passengers, but the gap to random shrinks as the groups grow, and none survives counting parties once. Consecutive runs come closest (p = 0.11 to 0.12 by party), consistent with the 2-to-10-apart correlation below.""",

"pair": """The correlation is concentrated on the same ticket (0.48). Across different tickets it is 0.15 at 2 to 3 apart and 0.07 at 4 to 10 apart, outside the random range in both cases and unchanged when same-party pairs are removed. 1 apart, and everything beyond 10 apart, sits inside the random range. The 3 pairs sharing a number under different prefixes (`751`, `752`, `17474`, `2673`) are too few to read; the 751 and 752 holders are 2nd and 3rd class passengers on different prefixes, not one group.""",

"funnel": """Most groups fall inside the funnel. The outliers that survive the adjustment for multiple testing are a handful of small groups: single-party tickets (1601, 347077) and a few runs that span several parties (111426 to 111428, 367226 to 367232, 364846 to 364851). The large windows sit close to zero, apart from 349xxx (53 train passengers, mean residual -0.11) and 175xx in the hundreds, both just outside the funnel and not significant after adjustment.""",
}
