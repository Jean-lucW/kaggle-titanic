# XGBoost with family survival features

This model is a gradient-boosted tree classifier (XGBoost). It predicts whether each test passenger survived, using passenger details plus what is known about how their relatives and ticket-mates fared.

## Features

Eleven features, in three groups.

- **Passenger details**: class, title (Mr, Mrs, Miss, Master, or Rare for the less common titles), sex, age, whether a cabin is recorded, and deck. Deck is the first letter of the cabin. Where the cabin is blank, it is taken from another passenger on the same ticket. Missing ages and decks are left missing, and XGBoost learns which way to send them at each split.
- **Family size**: siblings and spouses aboard (SibSp), and parents and children aboard (Parch).
- **Survival of people travelling together**: three features, each calculated only from train passengers whose outcome is known. A passenger's own outcome is never used in their own value.
  - Female-relative survival: the share of the passenger's mother, and for female passengers also their sisters and daughters, who survived.
  - Family survival: the share of all known relatives who survived.
  - Ticket-mate survival: the share of other passengers on the same ticket who survived, excluding the passenger's own family.

  Each is blank when no such person is in the train set.

Families were reconstructed beforehand from surnames, tickets, fares and the SibSp and Parch counts. Each passenger has a list of relatives with roles (parent, child, sibling, spouse).

## How the model was chosen

The 891 train passengers were split into four parts, keeping every family and ticket group within one part. Three of the parts served as holdouts. Each candidate model was trained three times, each time on 75% of the train passengers, and scored on the remaining 25%. The three holdouts do not overlap. Because relatives always fall in the same part, no model saw an outcome from its holdout through the survival features.

8 feature sets were tested against 445 hyperparameter settings (one-at-a-time sweeps plus 400 random combinations), for 10,680 models in total.

The main findings:

- Female-relative survival gave the largest gain. Adding it lifted median accuracy from 81.8% to 84.1%.
- Family survival added a little on top.
- Ticket-mate survival, fare and port of embarkation added nothing.
- Choosing the features mattered more than tuning the hyperparameters. Among the hyperparameters, the one that mattered most was the learning rate multiplied by the number of trees.

## Final model

- Features: all eleven above.
- Settings: trees up to 6 levels deep, learning rate 0.043, 800 trees, gamma 1.93, min_child_weight 1.49, 94% of rows and 44% of features sampled per tree, L2 penalty 0.11.

It scored 85.5% average accuracy on the three holdouts (87.0%, 87.4% and 82.1%) and had the lowest log loss of all the models tested. That score is the best of 400 tries on the same holdouts, so it is optimistic. Expect around 84% on unseen data.

The submitted model was retrained on all 891 train passengers. It predicts that 38.5% of test passengers survived.
