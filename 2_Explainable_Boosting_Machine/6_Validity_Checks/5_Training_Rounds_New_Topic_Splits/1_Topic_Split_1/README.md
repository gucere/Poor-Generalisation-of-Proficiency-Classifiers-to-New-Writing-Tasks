# Topic split 1

Part of [5_Training_Rounds_New_Topic_Splits](../README.md), an exploratory check added after the paper. Topic split 1 (`shuffle_1` in the code and the file names) is a random split of each course level's 8 writing topics into two halves of 4, drawn with the seed [42, 1]. The models of half a (`shuffle_1_a`) are trained on the training texts of the topics in the half-a column below, so the half-b topics are unseen for them; the models of half b (`shuffle_1_b`) the other way round. Every model is evaluated on all validation texts, separately for the topics it was trained on and those it was not.

## Topics

Topic numbers as in the topic map. [topic_split.json](topic_split.json) holds the same lists, and [topic_assignments.csv](../4_Analysis/topic_assignments.csv) all four splits side by side.

| Level | CEFR | Half a trains on (unseen for half b) | Half b trains on (unseen for half a) |
| --- | --- | --- | --- |
| 1 | A1 | 2, 4, 5, 7 | 1, 3, 6, 8 |
| 2 | A1 | 12, 14, 15, 16 | 9, 10, 11, 13 |
| 3 | A1 | 18, 20, 21, 23 | 17, 19, 22, 24 |
| 4 | A2 | 25, 26, 28, 30 | 27, 29, 31, 32 |
| 5 | A2 | 33, 34, 35, 40 | 36, 37, 38, 39 |
| 6 | A2 | 41, 42, 43, 45 | 44, 46, 47, 48 |
| 7 | B1 | 49, 52, 53, 56 | 50, 51, 54, 55 |
| 8 | B1 | 60, 62, 63, 64 | 57, 58, 59, 61 |
| 9 | B1 | 69, 70, 71, 72 | 65, 66, 67, 68 |
| 10 | B2 | 74, 75, 79, 80 | 73, 76, 77, 78 |
| 11 | B2 | 82, 84, 86, 87 | 81, 83, 85, 88 |
| 12 | B2 | 90, 91, 93, 95 | 89, 92, 94, 96 |
| 13 | C1 | 97, 99, 102, 103 | 98, 100, 101, 104 |
| 14 | C1 | 105, 106, 111, 112 | 107, 108, 109, 110 |
| 15 | C1 | 115, 116, 117, 119 | 113, 114, 118, 120 |

## Models

The main model (`combined_without_length_correlated_features`) as classifier and regressor, for each half, with 10, 25, 50, 100, 200 boosting rounds: 20 models, all other settings as in the paper. Each folder in `runs/` holds `config.json` (the settings, including the training topics), `completed.json` (training texts and minutes) and `validation_predictions.csv`; `logs/` holds the training logs. Both folders are derived from individual EFCAMDAT texts and are not in the public repository.

## Results

Means over 200 repeats; in each repeat, half of the validation learners choose the number of rounds by level macro F1 and the other half score the chosen model (see [4_Analysis](../4_Analysis/README.md)). Each value is the mean of the two halves. The last column is the share of repeats in which the rounds chosen on unseen topics did better.

| Model | Measure | Scored on | Rounds chosen on seen topics | Rounds chosen on unseen topics | 200 rounds | Unseen minus seen choice (2.5–97.5%) | Unseen choice better in |
| --- | --- | --- | --- | --- | --- | --- | --- |
| classifier | level macro F1 | unseen topics | 0.131 | 0.128 | 0.131 | -0.0027 (-0.0118 to +0.0034) | 22% |
| classifier | CEFR accuracy | unseen topics | 0.697 | 0.697 | 0.692 | -0.0001 (-0.0134 to +0.0123) | 49% |
| classifier | level macro F1 | seen topics | 0.949 | 0.944 | 0.950 | -0.0049 (-0.0164 to +0.0026) | 17% |
| classifier | CEFR accuracy | seen topics | 0.987 | 0.985 | 0.987 | -0.0013 (-0.0044 to +0.0005) | 24% |
| regressor | level macro F1 | unseen topics | 0.224 | 0.220 | 0.224 | -0.0037 (-0.0181 to +0.0054) | 21% |
| regressor | CEFR accuracy | unseen topics | 0.763 | 0.767 | 0.763 | +0.0032 (-0.0036 to +0.0129) | 72% |
| regressor | MAE (lower is better) | unseen topics | 1.050 | 1.046 | 1.050 | -0.0039 (-0.0174 to +0.0078) | 64% |
| regressor | level macro F1 | seen topics | 0.399 | 0.383 | 0.399 | -0.0162 (-0.0426 to +0.0000) | 0% |
| regressor | CEFR accuracy | seen topics | 0.874 | 0.868 | 0.874 | -0.0059 (-0.0172 to +0.0000) | 0% |
| regressor | MAE (lower is better) | seen topics | 0.546 | 0.566 | 0.546 | +0.0203 (+0.0000 to +0.0597) | 0% |
