# Project: Student Dropout dataset

**Student Dropout should work well as the project's main dataset.** The data is real, a clear experiment comes with it, and all the course models can run on it. That also removes the main risk of the hair CSV (`preguntes_projecte_csv.md`), which looks synthetic.

The project brief hasn't been reviewed, so this needs checking against its requirements.

## Dataset summary

- **Dataset:** [Predict Students' Dropout and Academic Success (UCI)](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success).
- **Source:** students on undergraduate degrees at one higher education institution, in fields such as nursing, management, design and technology. It was published with a [2021 paper](http://www.doi.org/10.1007/978-3-030-72657-7_16), which the authors ask users to cite.
- **Size:** 4,424 students and 36 features, with no missing values or duplicate rows.
- **Target:** the student's status at the end of the normal course length: Graduate (49.9%), Dropout (32.1%) or Enrolled (17.9%).
- **Features:**
  - 24 from the enrolment stage: academic path, demographics, family background, finances and economic indicators.
  - 6 from semester 1.
  - 6 from semester 2.

## Main question

**How early can we identify students at risk of dropping out?**

## First test results

Each stage adds more data, using logistic regression with 5-fold cross-validation and no tuning (2026-11-04).

| Data available | Accuracy (3 classes) | Macro F1 (3 classes) | Dropout AUC | Dropout recall / precision |
| --- | --- | --- | --- | --- |
| Enrolment, without finance variables | 60.1% | 0.49 | 0.75 | 66% / 52% |
| Enrolment | 65.2% | 0.55 | 0.82 | 70% / 61% |
| + Semester 1 | 74.9% | 0.66 | 0.91 | 80% / 75% |
| + Semester 2 | 77.9% | 0.71 | 0.92 | 82% / 78% |

Always predicting the largest class gives 49.9% accuracy and a macro F1 of 0.22. Recall and precision are for Dropout vs the rest, with balanced class weights.

Semester 1 adds far more than semester 2, which suggests the first semester is the key moment for spotting students at risk.

## Where it fits in the project

| Course topic | How it would be used |
| --- | --- |
| **Statistical learning** | Validation, confusion matrices, ROC and PR curves, leakage, and choosing a threshold |
| **Linear models** | Logistic regression as the baseline, with Lasso to find which variables matter |
| **k-NN** | Choosing `k`, feature scaling, and the effect of many one-hot columns |
| **Naive Bayes** | A model suited to discrete features, and a test of its independence assumption |
| **SVM** | Comparing linear and RBF kernels |

So far, logistic regression is best or tied at every stage, k-NN and Naive Bayes are behind, and the RBF kernel adds no clear gain. These models aren't tuned yet, so the ranking could change.

The main gap is regression, because this is a classification dataset. If the project needs it, the semester 2 grade could be predicted from earlier data, but this hasn't been tested.

## Issues to handle in the report

- **Finance variables:** 86.6% of students with unpaid fees drop out. UCI implies this is known at enrolment, but the fee status could be recorded later and reflect students who have already left. That's why the table shows enrolment results both ways.
- **Enrolled class:** these students have no final outcome yet, and the model finds only 17-39% of them. The main task should be Dropout vs the rest, with the three classes as a secondary result.
- **Coded categories:** course, parental occupation and similar variables are stored as numbers but are really categories. Treating them as categories raises macro F1 from 0.47 to 0.55 at the enrolment stage.
- **Zero grades:** a grade of 0 appears exactly as often as zero units passed (718 students in semester 1), so it probably means "no grade" rather than a real score.
- **Sensitive variables:** 45.1% of men drop out against 25.1% of women, but that doesn't show gender is the cause. Course and age may explain part of the gap, and using gender to flag students raises a fairness question worth discussing.
- **Generalization:** the data comes from one institution, so results may not transfer to others. Holding out groups with identical economic indicators, which are probably cohorts, barely lowers macro F1 (0.54 to 0.53). The indicators themselves add almost nothing.
