# Topic split 3

Part of [5_Training_Rounds_New_Topic_Splits](../README.md), an exploratory check added after the paper. Topic split 3 (`shuffle_3` in the code and the file names) is a random split of each course level's 8 writing topics into two halves of 4, drawn with the seed [42, 3]. The models of half a (`shuffle_3_a`) are trained on the training texts of the topics in the half-a column below, so the half-b topics are unseen for them; the models of half b (`shuffle_3_b`) the other way round. Every model is evaluated on all validation texts, separately for the topics it was trained on and those it was not.

## Topics

Topic numbers as in the topic map. [topic_split.json](topic_split.json) holds the same lists, and [topic_assignments.csv](../4_Analysis/topic_assignments.csv) all four splits side by side.

| Level | CEFR | Half a trains on (unseen for half b) | Half b trains on (unseen for half a) |
| --- | --- | --- | --- |
| 1 | A1 | 2, 4, 5, 8 | 1, 3, 6, 7 |
| 2 | A1 | 11, 13, 14, 16 | 9, 10, 12, 15 |
| 3 | A1 | 17, 19, 20, 23 | 18, 21, 22, 24 |
| 4 | A2 | 25, 28, 29, 31 | 26, 27, 30, 32 |
| 5 | A2 | 34, 35, 37, 39 | 33, 36, 38, 40 |
| 6 | A2 | 44, 46, 47, 48 | 41, 42, 43, 45 |
| 7 | B1 | 50, 53, 54, 55 | 49, 51, 52, 56 |
| 8 | B1 | 57, 58, 62, 63 | 59, 60, 61, 64 |
| 9 | B1 | 65, 66, 68, 71 | 67, 69, 70, 72 |
| 10 | B2 | 73, 76, 77, 80 | 74, 75, 78, 79 |
| 11 | B2 | 81, 83, 85, 87 | 82, 84, 86, 88 |
| 12 | B2 | 89, 91, 92, 94 | 90, 93, 95, 96 |
| 13 | C1 | 98, 101, 102, 103 | 97, 99, 100, 104 |
| 14 | C1 | 105, 108, 109, 110 | 106, 107, 111, 112 |
| 15 | C1 | 114, 115, 116, 118 | 113, 117, 119, 120 |

## Models

The main model (`combined_without_length_correlated_features`) as classifier and regressor, for each half, with 10, 25, 50, 100, 200 boosting rounds: 20 models, all other settings as in the paper. Each folder in `runs/` holds `config.json` (the settings, including the training topics), `completed.json` (training texts and minutes) and `validation_predictions.csv`; `logs/` holds the training logs. Both folders are derived from individual EFCAMDAT texts and are not in the public repository.

## Results

Means over 200 repeats; in each repeat, half of the validation learners choose the number of rounds by level macro F1 and the other half score the chosen model (see [4_Analysis](../4_Analysis/README.md)). Each value is the mean of the two halves. The last column is the share of repeats in which the rounds chosen on unseen topics did better.

| Model | Measure | Scored on | Rounds chosen on seen topics | Rounds chosen on unseen topics | 200 rounds | Unseen minus seen choice (2.5–97.5%) | Unseen choice better in |
| --- | --- | --- | --- | --- | --- | --- | --- |
| classifier | level macro F1 | unseen topics | 0.210 | 0.210 | 0.209 | +0.0006 (-0.0080 to +0.0088) | 54% |
| classifier | CEFR accuracy | unseen topics | 0.693 | 0.695 | 0.691 | +0.0023 (-0.0075 to +0.0124) | 68% |
| classifier | level macro F1 | seen topics | 0.944 | 0.942 | 0.945 | -0.0012 (-0.0075 to +0.0047) | 34% |
| classifier | CEFR accuracy | seen topics | 0.987 | 0.986 | 0.986 | -0.0008 (-0.0024 to +0.0008) | 21% |
| regressor | level macro F1 | unseen topics | 0.232 | 0.230 | 0.232 | -0.0024 (-0.0142 to +0.0053) | 28% |
| regressor | CEFR accuracy | unseen topics | 0.776 | 0.777 | 0.776 | +0.0010 (-0.0050 to +0.0072) | 62% |
| regressor | MAE (lower is better) | unseen topics | 0.994 | 0.990 | 0.994 | -0.0044 (-0.0182 to +0.0107) | 80% |
| regressor | level macro F1 | seen topics | 0.398 | 0.381 | 0.398 | -0.0174 (-0.0531 to +0.0000) | 0% |
| regressor | CEFR accuracy | seen topics | 0.871 | 0.866 | 0.871 | -0.0053 (-0.0183 to +0.0000) | 0% |
| regressor | MAE (lower is better) | seen topics | 0.556 | 0.580 | 0.556 | +0.0235 (+0.0000 to +0.0679) | 0% |
