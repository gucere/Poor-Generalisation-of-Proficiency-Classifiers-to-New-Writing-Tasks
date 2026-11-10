# Training Rounds on New Topic Splits

An exploratory check added after the paper (November 2026). [4_Training_Rounds](../4_Training_Rounds) compared two ways of choosing the number of boosting rounds on one topic split only, the topic check's. This folder repeats that comparison on three new random topic splits, to see whether the result depends on which topics the models were trained on.

The numbers come from the validation split only. The test set is not used, and the paper does not include this check.

## Question

A model is trained on some writing topics of every course level and is meant to work on new topics. Should its number of boosting rounds be chosen on validation texts of the topics it was trained on (seen topics), or of topics it never saw (unseen topics)? The second should favour a model that transfers to new topics. Does it, and does the answer change with the topic split?

The code and its column names call the topics "tasks" (`seen_tasks`, `unseen_tasks`): every writing topic is one EFCAMDAT writing task.

## Design

- **Topic splits.** Every one of the 15 course levels has 8 writing topics. A topic split divides each level's 8 topics into two halves of 4. Four splits are compared:
  - **The topic check's split** (`section_3` in the code): the split of the paper's topic check, called the half-task split in the repository README ([1_Topic_Check](../1_Topic_Check)), with the topics of `topic_holdout.json` (not in this repository) as one half (`section_3_held_out`) and the rest as the other (`section_3_seen`). The code calls it `section_3` because these topics were first held out from the LLM fine-tuning in section 3 (LLM prediction) of the thesis project this code comes from. It is not a section of the paper.
  - **Topic splits 1–3** (`shuffle_1` to `shuffle_3`): new random splits of each level's topics, drawn with the seeds [42, 1], [42, 2] and [42, 3].
  - Within a level, a half of a new split shares 2 or 3 of its 4 topics with a half of the topic check's split. The one exception is level 5 of split 2, which repeats the topic check's halves.
- **Models.** For each half of each split, the main model (`combined_without_length_correlated_features`) is trained as classifier and regressor on that half's training texts, with 10, 25, 50, 100 and 200 boosting rounds.
  - Each half has 135,074–149,678 of the 284,752 training texts.
  - All other settings are those of the paper: fixed rounds, no early stopping, and the validation texts are never used in training.
  - That is 20 models per new split, so 60 new models. The topic check's models are those of [1_Topic_Check](../1_Topic_Check) (200 rounds) and [4_Training_Rounds](../4_Training_Rounds) (10–100 rounds).
- **Evaluation.** Each model predicts all 60,230 validation texts, whose learners are not in the training split. For a model, the topics of its own half are seen and the other half's topics are unseen.
- **Selection comparison.** 200 repeats with seed 42. In each repeat:
  - a random half of the validation learners choose the number of rounds and the other half score it, so no learner does both;
  - each level's 4 unseen topics are split at random into 2 for choosing and 2 for scoring;
  - the number of rounds is chosen by level macro F1, once on the choosing learners' texts of the seen topics (**seen choice**) and once on their texts of the 2 choosing unseen topics (**unseen choice**). Ties go to more rounds;
  - both choices, and the paper's fixed **200 rounds**, are scored on the scoring learners' texts of the 2 scoring unseen topics, and of the seen topics.
- **Shared draws.** In each repeat, all four splits use the same halves of the validation learners. The topic check split's draws of choosing topics come from the same random stream as the learner halves, and every new split has a stream of its own. So its numbers here are exactly those of [4_Training_Rounds](../4_Training_Rounds).
- **Exploratory.** The validation data had already been examined before this check was designed.

## Which Topics Are in Which Half

Topic numbers, as in the topic map. Each cell lists the 4 topics of one half; the level's other 4 topics form the other half. In the topic check's split, the listed half holds the held-out topics of `topic_holdout.json`. In the new splits it is half a. The models of a half are trained on its topics, so the other half's topics are unseen for them.

| Level | CEFR | Topic check (held-out topics) | Split 1 (half a) | Split 2 (half a) | Split 3 (half a) |
| --- | --- | --- | --- | --- | --- |
| 1 | A1 | 2, 3, 5, 8 | 2, 4, 5, 7 | 2, 6, 7, 8 | 2, 4, 5, 8 |
| 2 | A1 | 12, 13, 14, 15 | 12, 14, 15, 16 | 10, 11, 14, 16 | 11, 13, 14, 16 |
| 3 | A1 | 19, 20, 23, 24 | 18, 20, 21, 23 | 17, 21, 23, 24 | 17, 19, 20, 23 |
| 4 | A2 | 25, 27, 28, 30 | 25, 26, 28, 30 | 25, 28, 29, 31 | 25, 28, 29, 31 |
| 5 | A2 | 34, 37, 39, 40 | 33, 34, 35, 40 | 34, 37, 39, 40 | 34, 35, 37, 39 |
| 6 | A2 | 42, 43, 45, 48 | 41, 42, 43, 45 | 41, 44, 45, 46 | 44, 46, 47, 48 |
| 7 | B1 | 51, 53, 55, 56 | 49, 52, 53, 56 | 49, 50, 51, 55 | 50, 53, 54, 55 |
| 8 | B1 | 57, 58, 59, 62 | 60, 62, 63, 64 | 57, 58, 61, 62 | 57, 58, 62, 63 |
| 9 | B1 | 66, 68, 69, 71 | 69, 70, 71, 72 | 66, 68, 70, 71 | 65, 66, 68, 71 |
| 10 | B2 | 74, 78, 79, 80 | 74, 75, 79, 80 | 73, 74, 76, 79 | 73, 76, 77, 80 |
| 11 | B2 | 81, 82, 83, 85 | 82, 84, 86, 87 | 81, 82, 87, 88 | 81, 83, 85, 87 |
| 12 | B2 | 90, 91, 93, 94 | 90, 91, 93, 95 | 89, 91, 92, 93 | 89, 91, 92, 94 |
| 13 | C1 | 97, 101, 102, 103 | 97, 99, 102, 103 | 98, 101, 103, 104 | 98, 101, 102, 103 |
| 14 | C1 | 106, 108, 109, 111 | 105, 106, 111, 112 | 105, 107, 111, 112 | 105, 108, 109, 110 |
| 15 | C1 | 113, 115, 116, 118 | 115, 116, 117, 119 | 113, 114, 116, 120 | 114, 115, 116, 118 |

These lists can be found in three places:
- every split folder's README, with both halves;
- its `topic_split.json`;
- [4_Analysis/topic_assignments.csv](4_Analysis/topic_assignments.csv), one row per topic with its half in every split.

## Results

**Level macro F1 on unseen topics**, the measure the rounds are chosen by. These are means over the 200 repeats, each the mean of the split's two halves; "all four" is the mean of all eight halves. The last column is the share of repeats in which the unseen choice did better.

| Split | Model | Seen choice | Unseen choice | 200 rounds | Unseen minus seen choice | Unseen choice better in |
| --- | --- | --- | --- | --- | --- | --- |
| Topic check | classifier | 0.170 | 0.168 | 0.171 | −0.0013 | 37% |
| Split 1 | classifier | 0.131 | 0.128 | 0.131 | −0.0027 | 22% |
| Split 2 | classifier | 0.157 | 0.155 | 0.157 | −0.0026 | 16% |
| Split 3 | classifier | 0.210 | 0.210 | 0.209 | +0.0006 | 54% |
| All four | classifier | 0.167 | 0.165 | 0.167 | −0.0015 | 24% |
| Topic check | regressor | 0.231 | 0.227 | 0.231 | −0.0035 | 18% |
| Split 1 | regressor | 0.224 | 0.220 | 0.224 | −0.0037 | 21% |
| Split 2 | regressor | 0.192 | 0.195 | 0.192 | +0.0027 | 70% |
| Split 3 | regressor | 0.232 | 0.230 | 0.232 | −0.0024 | 28% |
| All four | regressor | 0.220 | 0.218 | 0.220 | −0.0017 | 30% |

**The regressor's CEFR accuracy and MAE on unseen topics**, with the rounds still chosen by level macro F1:

| Split | CEFR accuracy: seen choice → unseen choice | Unseen choice better in | MAE: seen choice → unseen choice | Unseen choice better in |
| --- | --- | --- | --- | --- |
| Topic check | 0.750 → 0.756 | 84% | 1.037 → 1.026 | 82% |
| Split 1 | 0.763 → 0.767 | 72% | 1.050 → 1.046 | 64% |
| Split 2 | 0.734 → 0.748 | 97% | 1.204 → 1.166 | 99% |
| Split 3 | 0.776 → 0.777 | 62% | 0.994 → 0.990 | 80% |
| All four | 0.756 → 0.762 | 99% | 1.071 → 1.057 | 99.5% |

- **Classifier: choosing on unseen topics does not help.** It is slightly worse than choosing on seen topics in three of the four splits and level with it in split 3. The fixed 200 rounds is as good as either choice in every split.
- **Regressor: a small trade.**
  - Choosing on unseen topics picks fewer rounds, which lowers level macro F1 in three splits.
  - It raises CEFR accuracy and lowers MAE on unseen topics in every split. The gain is clear in split 2 and small in the others.
  - On seen topics it costs more than it gains: level macro F1 falls by 0.024 and MAE rises by 0.031, over all four splits.
- **The unseen-topic choice is unstable.**
  - Chosen on seen topics, the classifier's rounds are mostly 50 or 200 (38% and 43% over all four splits), and the regressor's are almost always 200 (99.6%).
  - Chosen on unseen topics, every number from 10 to 200 is picked in 11–32% of the cases, because the unseen-topic curves are nearly flat ([4_Analysis/curve.png](4_Analysis/curve.png)).
- **The topic split matters more than the number of rounds.**
  - Over all validation texts of unseen topics, the classifier with 200 rounds reaches level macro F1 0.176, 0.133, 0.158 and 0.214 in the topic check's split and splits 1–3.
  - The number of rounds moves it by 0.003–0.015 within a split. In splits 2 and 3, fewer rounds make it worse, not better.
  - The regressor's MAE with 200 rounds ranges from 1.00 (split 3) to 1.21 (split 2).
  - On seen topics, every split reaches level macro F1 0.94–0.95 with 200 rounds. So the gap between seen and unseen topics that the paper's topic check reports does not depend on the topic split or on the training length.

**In short.** The conclusion from the topic check's split holds on three new topic splits: choosing the number of rounds on unseen topics does not give a classifier that transfers better to new topics. For the regressor, it trades accuracy on seen topics for a small gain in CEFR accuracy and MAE on unseen topics.

**What the percentages do not measure.** The "better in" shares and the 2.5–97.5% ranges in the tables of [4_Analysis](4_Analysis) only reflect which validation learners and topics are drawn in each repeat. The models are fixed, so the shares do not include the variation from training. They are not significance tests.

## Files

| Path | Content |
| --- | --- |
| [1_Topic_Split_1](1_Topic_Split_1), [2_Topic_Split_2](2_Topic_Split_2), [3_Topic_Split_3](3_Topic_Split_3) | One folder per new split. `README.md`: both halves' topics and the split's results. `topic_split.json`: the same topic lists and the seed. The 20 models (`runs/`) and their training logs (`logs/`) are not in this repository. |
| [4_Analysis](4_Analysis) | The comparison over all four splits: tables, figures and topic assignments (see its [README](4_Analysis/README.md)). |

The models and logs are derived from individual EFCAMDAT texts, and the topic names are course material, so this copy has the topic numbers only.

## How to Run

From the repository folder:

```powershell
python "2_Explainable_Boosting_Machine/6_Validity_Checks/validity_checks.py" rounds --splits shuffle_1 --trained-on shuffle_1_a
python "2_Explainable_Boosting_Machine/6_Validity_Checks/validity_checks.py" rounds --splits shuffle_1 --trained-on shuffle_1_b
python "2_Explainable_Boosting_Machine/6_Validity_Checks/validity_checks.py" splits
```

- **Training (`rounds`).** It trains the 10 models of one half; the other halves and splits follow the same pattern (`shuffle_2_a`, …, `shuffle_3_b`). Without `--trained-on`, one process trains both halves of the given `--splits`. Models that already exist are not trained again.
- **Analysis (`splits`).** It trains nothing: it reads the saved predictions of all four splits. It writes `4_Analysis`, every split folder's `README.md` and `topic_split.json`.
  - `--no-topic-names` leaves out the topic names; this repository's copies were written with it.
  - `--repeats` and `--select-by` change the number of repeats and the measure the rounds are chosen by.
- **The topic check's split only (`selection`).** It writes the same comparison for that split alone to `4_Training_Rounds/selection_by_topic_*.csv`.
