# Findings text for tickets_model_v1.ipynb, read by build_tickets_model_v1.py. Numbers are from the run on complete_data.csv.

FINDINGS = {
"summary": """## Findings

- **Yes, for logistic regression, by about 2 points.** Adding the ticket number window lifts cross-validated accuracy from 79.6% to 81.8% (+2.2 points, p = 0.01). All ticket features together give 81.9% (+2.4, p = 0.01). Ticket type alone adds 1.1 points (p = 0.03).
- **The number does the work, not the prefix or the group size.** Band and window beat type, and ticket group size adds nothing (-0.0 points).
- **The gain is modest and model-dependent.** AUC rises by 0.011 at most, and linear regression gains under 1 point with any ticket set. The ticket mostly repeats class and port, which the model already has.
- **What the window adds looks like shared fate, not the number range.** The windows with the largest weights are mostly a few families on consecutive tickets (367xxx, 230xxx, PC 174xx, 349xxx). A test passenger on a ticket shared with train passengers inherits their outcome. A family or ticket-group survival feature would likely capture the same signal more directly.
- **Deck.** The exact ticket identifies the deck: passengers on the same ticket share it 98% of the time. The ticket number does not: a neighbouring ticket guesses the deck right 36% of the time, against 32% for always guessing C. The `Deck` column added to the data files uses the passenger's own cabin, then ticket-mates' cabins.""",

"cv": """Logistic regression gains 1.8 to 2.4 points from band, window or all ticket features, each with p of 0.03 or less. Log loss falls from 0.448 to 0.436 with the window, so the probabilities also improve. Linear regression starts slightly higher (79.9%) and gains at most 0.8 points: with straight-line fitted values on a 0/1 outcome, the many small window columns mostly add noise.""",

"chart": """Every ticket set except group size clears 0 on accuracy, but the intervals are wide: the window's +2.2 points could be anywhere from about +0.5 to +3.9. On AUC only the window clears 0. A gain of 1 to 3 points is worth keeping in a model but not worth expecting to hold exactly on the test set.""",

"coef": """The strongest ticket columns are small windows: 367xxx (9 train passengers, 67% survived), 16xx (7), PC 174xx (10, 90% survived), 230xxx (8, 88% survived). The one large window with a strong weight is 349xxx (53 train passengers, 6% survived against 19% expected from class and sex). Sex and class still carry the largest weights. Because the strong windows are small and family-shaped, the ticket gain is likely the same signal as family survival.""",
}
