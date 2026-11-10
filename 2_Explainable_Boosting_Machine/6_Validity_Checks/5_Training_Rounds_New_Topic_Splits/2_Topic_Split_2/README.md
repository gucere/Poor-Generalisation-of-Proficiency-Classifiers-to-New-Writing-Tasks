# Topic split 2

Part of [5_Training_Rounds_New_Topic_Splits](../README.md), an exploratory check added after the paper. Topic split 2 (`shuffle_2` in the code and the file names) is a random split of each course level's 8 writing topics into two halves of 4, drawn with the seed [42, 2]. The models of half a (`shuffle_2_a`) are trained on the training texts of the topics in the half-a column below, so the half-b topics are unseen for them; the models of half b (`shuffle_2_b`) the other way round. Every model is evaluated on all validation texts, separately for the topics it was trained on and those it was not.

## Topics

Topic numbers as in the topic map. [topic_split.json](topic_split.json) holds the same lists, and [topic_assignments.csv](../4_Analysis/topic_assignments.csv) all four splits side by side.

| Level | CEFR | Half a trains on (unseen for half b) | Half b trains on (unseen for half a) |
| --- | --- | --- | --- |
| 1 | A1 | 2, 6, 7, 8 | 1, 3, 4, 5 |
| 2 | A1 | 10, 11, 14, 16 | 9, 12, 13, 15 |
| 3 | A1 | 17, 21, 23, 24 | 18, 19, 20, 22 |
| 4 | A2 | 25, 28, 29, 31 | 26, 27, 30, 32 |
| 5 | A2 | 34, 37, 39, 40 | 33, 35, 36, 38 |
| 6 | A2 | 41, 44, 45, 46 | 42, 43, 47, 48 |
| 7 | B1 | 49, 50, 51, 55 | 52, 53, 54, 56 |
| 8 | B1 | 57, 58, 61, 62 | 59, 60, 63, 64 |
| 9 | B1 | 66, 68, 70, 71 | 65, 67, 69, 72 |
| 10 | B2 | 73, 74, 76, 79 | 75, 77, 78, 80 |
| 11 | B2 | 81, 82, 87, 88 | 83, 84, 85, 86 |
| 12 | B2 | 89, 91, 92, 93 | 90, 94, 95, 96 |
| 13 | C1 | 98, 101, 103, 104 | 97, 99, 100, 102 |
| 14 | C1 | 105, 107, 111, 112 | 106, 108, 109, 110 |
| 15 | C1 | 113, 114, 116, 120 | 115, 117, 118, 119 |

## Models

The main model (`combined_without_length_correlated_features`) as classifier and regressor, for each half, with 10, 25, 50, 100, 200 boosting rounds: 20 models, all other settings as in the paper. Each folder in `runs/` holds `config.json` (the settings, including the training topics), `completed.json` (training texts and minutes) and `validation_predictions.csv`; `logs/` holds the training logs. Both folders are derived from individual EFCAMDAT texts and are not in the public repository.

## Results

Means over 200 repeats; in each repeat, half of the validation learners choose the number of rounds by level macro F1 and the other half score the chosen model (see [4_Analysis](../4_Analysis/README.md)). Each value is the mean of the two halves. The last column is the share of repeats in which the rounds chosen on unseen topics did better.

| Model | Measure | Scored on | Rounds chosen on seen topics | Rounds chosen on unseen topics | 200 rounds | Unseen minus seen choice (2.5–97.5%) | Unseen choice better in |
| --- | --- | --- | --- | --- | --- | --- | --- |
| classifier | level macro F1 | unseen topics | 0.157 | 0.155 | 0.157 | -0.0026 (-0.0145 to +0.0044) | 16% |
| classifier | CEFR accuracy | unseen topics | 0.669 | 0.668 | 0.667 | -0.0004 (-0.0102 to +0.0071) | 36% |
| classifier | level macro F1 | seen topics | 0.947 | 0.947 | 0.948 | +0.0003 (-0.0074 to +0.0059) | 52% |
| classifier | CEFR accuracy | seen topics | 0.987 | 0.987 | 0.987 | -0.0002 (-0.0023 to +0.0006) | 31% |
| regressor | level macro F1 | unseen topics | 0.192 | 0.195 | 0.192 | +0.0027 (-0.0068 to +0.0139) | 70% |
| regressor | CEFR accuracy | unseen topics | 0.734 | 0.748 | 0.734 | +0.0140 (+0.0000 to +0.0303) | 97% |
| regressor | MAE (lower is better) | unseen topics | 1.204 | 1.166 | 1.204 | -0.0383 (-0.0704 to -0.0047) | 99% |
| regressor | level macro F1 | seen topics | 0.405 | 0.367 | 0.405 | -0.0379 (-0.0638 to -0.0061) | 0% |
| regressor | CEFR accuracy | seen topics | 0.871 | 0.858 | 0.871 | -0.0127 (-0.0232 to -0.0015) | 0% |
| regressor | MAE (lower is better) | seen topics | 0.538 | 0.586 | 0.538 | +0.0472 (+0.0046 to +0.0841) | 0% |
