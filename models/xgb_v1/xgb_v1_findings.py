# Findings text for xgb_v1.ipynb, read by build_xgb_v1.py. Numbers are from the run on complete_data.csv.

FINDINGS = {
"summary": """## Findings

- **Features matter far more than hyperparameters.** Moving from the worst to the best feature set improves median log loss from 0.447 to 0.400. Across the 400 random settings, most of the gap between good and typical hyperparameters is about 0.02.
- **Female-relative survival is the feature that counts.** Added to Core + SibSp and Parch, it lifts median accuracy from 81.8% to 84.1% and beats that set in 94% of the 400 settings. Family survival adds 1.5 points. Combining the two is best on log loss (best in 40% of settings, average rank 1.9 of 8).
- **Ticket-mate survival and fare plus port add nothing.** Ticket-mate survival alone ties Core + SibSp and Parch (better in 49% of settings). Adding it, fare and port to the other two survival features leaves accuracy unchanged.
- **SibSp and Parch help.** Core alone is 1.9 points worse on median accuracy, and worse on log loss in 94% of settings.
- **Hyperparameters: learn enough, but not too much.** Learning rate × number of trees is the setting that matters most: below 2 the model underfits badly (median log loss 0.48). Between 5 and 50 it levels off. Trees 2 to 6 deep do equally well; depth 1 and depth 8 are worse. The other hyperparameters move log loss by about 0.01.
- **Use about 85% as the expected accuracy, not the best score.** The best tuned models score 85.2% to 85.5%, but that is the best of 400 tries on the same three holdouts. Holdout 3 scores about 3 points below holdouts 1 and 2 for every feature set, which is larger than most of the differences between hyperparameter settings.""",

"features": """The order is the same at the baseline, at the median and at the best setting: the sets with female-relative survival lead, Core trails. Each survival set beats Core + SibSp and Parch on log loss in 95% to 99% of the 400 settings, so the gain does not depend on the hyperparameters. The top four sets are within 0.5 points of median accuracy of each other, which is about 3 passengers across the three holdouts. On accuracy alone they cannot be separated.""",

"sweep": """With the survival features in (blue), most of the curves are nearly flat: no single hyperparameter moves accuracy by more than about 1 point from the baseline. Log loss shows the pattern more clearly. It gets worse as trees grow deeper than 2, as the learning rate rises above 0.05 and as trees go past 300, all signs of overfitting. Stronger L2 regularisation (lambda 3 to 10) and a small gamma (around 1 to 2) help a little. Core and Core + SibSp and Parch are more sensitive to these settings than the set with survival features.""",

"random": """The top 15 is dominated by sets with female-relative survival, at a range of different settings: depth 2 to 6 with few trees, or many trees with a low learning rate. Many different hyperparameter settings reach the same score. Across all settings, the clearest pattern is the product of learning rate and number of trees: too little learning costs far more than too much. The best 10% of settings for the + all three survival set use a lower min_child_weight (median 1.2, against 2.9 for all settings) and otherwise look like the rest.""",

"importance": """Title carries the most information. It holds sex and age group together: Master marks a boy. Female-relative survival is second, ahead of sex, and family survival is fourth. Ticket-mate survival, SibSp and Parch contribute almost nothing once the survival features are in. SibSp and Parch helped the Core set because they partly stand in for family, which the survival features now cover directly.""",
}
