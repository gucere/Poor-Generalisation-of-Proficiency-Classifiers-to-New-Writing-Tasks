# Analysis of the Four Topic Splits

The selection comparison over the topic check's split and topic splits 1–3, written by `validity_checks.py splits` from the saved validation predictions of the 80 models (20 per split). The [folder README](../README.md) explains the design and summarises the results.

In the files:
- `split` is `section_3` (the topic check's split), `shuffle_1`, `shuffle_2` or `shuffle_3` (topic splits 1–3), or `all` for all four together;
- "tasks" are writing topics (`seen_tasks` are the topics a model was trained on, `unseen_tasks` the others);
- the two halves of a split are `trained_on` `section_3_held_out` and `section_3_seen`, or `shuffle_k_a` and `shuffle_k_b`.

## Figures

| File | Shows |
| --- | --- |
| `curve.png`, `curve.pdf` | Level macro F1 and CEFR accuracy by number of boosting rounds, for the main classifier and regressor, on the seen (solid) and unseen (dashed) topics of every split. Each line is the mean of the split's two halves, over all validation texts of those topics. |
| `choice_difference.png`, `choice_difference.pdf` | Level macro F1 on unseen scoring topics: the unseen choice minus the seen choice, per split and for all four, with the 2.5–97.5% range over the 200 repeats. A point right of the zero line means choosing on unseen topics did better. |

## Tables

| File | One row per | Columns |
| --- | --- | --- |
| `summary.csv` | split, model, scored topics and measure | **Means over the repeats.** `seen_choice`, `unseen_choice`, `200_rounds`: the mean score of each approach, each repeat being the mean over the split's halves (over all eight halves for `all`). `unseen_minus_seen_choice`, `difference_2.5%`, `difference_97.5%`: the mean difference and its range over the repeats. `share_unseen_choice_better`: how often the unseen choice did better (lower is better for `regression_mae`). `share_equal`: how often they tied. `share_same_rounds`: how often both chose the same number of rounds. `repeats`. `scored_on` is `unseen_tasks` (the scoring learners' texts of the 2 scoring unseen topics per level) or `seen_tasks` (their texts of the seen topics). |
| `chosen_rounds.csv` | split, model, choice and number of rounds | `choices`: how often that number of rounds was chosen, over repeats and halves. `share`: the same as a share. |
| `curve.csv` | split, model, number of rounds and topic group | Level macro F1 and accuracy, CEFR macro F1 and accuracy, and, for the regressor, MAE in levels. These are on all validation texts of the seen or unseen topics, as in `curve.png`, not only the scoring learners'. |
| `repeats.csv` | repeat, split, model and half | `seen_choice_rounds`, `unseen_choice_rounds`: the rounds each approach chose. `choosing_topics`: the 2 unseen topics per level used for choosing. `texts_*`: the number of texts in each subset. The scores of every approach on both topic groups (`<measure>_<scored_on>_<approach>`). |
| `topic_assignments.csv` | topic | `level`, `topic_id` and, per split, the half that is trained on the topic. The topic names are course material and not in this repository. For `section_3` the halves are `held_out` (the topics of `0_Topics/topic_holdout.json`) and `seen`; for the new splits they are `a` and `b`. |

**Choosing.** The number of rounds is chosen by level macro F1, and ties go to more rounds. Every repeat (seed 42) draws a new half of the validation learners, which is the same for all splits, and new choosing topics.

**What the ranges mean.** The 2.5–97.5% ranges and the shares describe the variation over these draws. They do not include the variation from training the models, so they are not significance tests.
