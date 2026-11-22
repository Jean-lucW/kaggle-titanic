# Findings text for tickets_groups_v1.ipynb, read by build_tickets_groups_v1.py. Numbers are from the run on complete_data.csv.

FINDINGS = {
"summary": """## Findings

- **Ticket and family are mostly the same groups.** 73% of same-ticket pairs are family, and once family survival is in the model the ticket adds nothing further (+1.0 points for family, +0.9 for ticket or family).
- **Group survival helps a little, but not detectably.** Family, ticket, cabin and nearby-ticket survival rates each add 0.3 to 1.0 points of accuracy, none significant (p from 0.26 to 0.43). Ticket or family plus nearby tickets together add 1.6 points (p = 0.07): the best candidate, still borderline.
- **Nearby ticket numbers are class and port.** Non-family tickets 2 to 10 apart share a class 99% of the time and a port 89%. Matched on class, port and sex, their survival correlates only 0.02 more than tickets far apart.
- **Class, fare, ticket band and cabin recorded largely carry the same information** (Cramér's V 0.62 to 0.85 between class and the others). Port is only moderately tied to them (0.28 to 0.51).""",

"q1": """Most ticket groups are families, but not all: 96 same-ticket pairs are not family, and 75 train passengers share a ticket while travelling with no family. That is the only thing the ticket can add over family.""",

"q2a": """Members of the same ticket and family share survival most strongly (0.55), and non-family ticket-mates still share it (0.34). Cabin-mates share less. The bottom two bars are the key comparison: strangers of the same class, port and sex correlate at 0.28 even on far-apart tickets, so most of the nearby-ticket correlation is class, port and sex, not the number.""",

"q2b": """Every interval crosses 0. Family is the single most useful group feature (+1.0 points), and ticket adds nothing on top of it. Cabin is limited by coverage: only 12% of train passengers have a cabin-mate in train. Nearby tickets cover 68% of passengers but carry little beyond class and port. The combination of ticket or family with nearby tickets (+1.6, p = 0.07) is worth re-testing in the final model, not banking on.""",

"q3": """Nearby ticket numbers are almost always the same class (99% against 40% for random pairs) and usually the same port (89% against 57%). They are no more likely to share a sex (56% against 54%), so they are not mainly couples or families.""",

"q4": """Class is tied most strongly to ticket band (0.85) and to whether a cabin is recorded (0.78), then to fare (0.63). Port is the most independent of the five, linked mainly through the ticket band (0.51), because some number ranges were sold at one port (26xx at Cherbourg, 330xxx and 367xxx at Queenstown). For a linear model, this means class, fare, ticket band and cabin recorded overlap heavily: adding all of them mostly splits the same effect between columns.""",
}
