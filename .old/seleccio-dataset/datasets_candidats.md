# Project: candidate datasets (UCI)

Twenty non-housing datasets from the UCI repository, sorted by whether they meet the project's requirement.

**Requirement:** the dataset's creators must not have published a paper doing the same analysis the project wants to do. Papers by other researchers are fine, and nearly every public dataset has some.

**Recommendation: AIDS Clinical Trials Group Study 175 (#10).** Its creators ran a clinical trial comparing treatments and published no machine learning work, so predicting patient outcomes would be the project's own question.

Only 3 of the 20 clearly meet the requirement. For more options, see `datasets_other_sources.md`.

## How this list was checked

- **Assumption:** the project brief was not available, so this list assumes a tabular dataset that works with the course models (linear models, k-NN, Naive Bayes, SVM, decision trees and random forests).
- **Checked (2026-11-04 and 2026-11-09):** links, row counts and feature counts come from UCI's metadata. Each dataset's linked paper was compared with the project's question. For the remaining candidates, Seoul Bike and Myocardial Infarction, the creators' other publications were also searched.
- **Not tested:** the "watch out for" notes come from general knowledge and search results. Only Student Dropout has been downloaded and modelled (`student_dropout_summary.md`).

## Remaining candidates

| # | Dataset | Rows / features | What the creators did | Question to answer | Watch out for |
| --- | --- | --- | --- | --- | --- |
| 10 | [AIDS Clinical Trials Group Study 175](https://archive.ics.uci.edu/dataset/890/aids+clinical+trials+group+study+175) | 2,139 / 23 | A 1996 trial comparing single-drug and combination treatments; no machine learning | Which patients' disease will progress, and from which data? | The target is a combined event (immune decline, AIDS or death), not just death, and some patients' outcomes are unknown because follow-up ended. The follow-up time leaks the outcome. |
| 19 | [Communities and Crime](https://archive.ics.uci.edu/dataset/183/communities+and+crime) | 1,994 / 127 | Built a software tool to help police departments share information | Which social factors predict violent crime? | Its target is a number, so it also covers regression. Many police variables are missing, and race-related variables raise fairness issues. |
| 8 | [Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) | 101,766 / 47 | Studied whether measuring HbA1c affects readmission, using logistic regression | Can we predict 30-day readmission, and which model works best? | Partial overlap: the creators also used readmission as their target, to measure one effect rather than to predict. Patients repeat, so split by patient. The size needs subsampling for SVM and k-NN. |

## Not fully checked

| # | Dataset | Rows / features | What is known | Question to answer | Watch out for |
| --- | --- | --- | --- | --- | --- |
| 3 | [In-Vehicle Coupon Recommendation](https://archive.ics.uci.edu/dataset/603/in+vehicle+coupon+recommendation) | 12,684 / 25 | The creators' paper presents a new rule-learning method, using coupon acceptance as an example: partial overlap | Which drivers accept a coupon, and in what context? | Many categories create a lot of one-hot columns |
| 9 | [CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) | 253,680 / 21 | UCI links an unrelated CDC report; not searched further | How well do survey answers flag diabetes? | Too large for kernel SVM or k-NN without subsampling |
| 15 | [Spambase](https://archive.ics.uci.edu/dataset/94/spambase) | 4,601 / 57 | UCI links no paper; not searched further | Which word patterns identify spam? | Old data, and a very common project |
| 17 | [Steel Plates Faults](https://archive.ics.uci.edu/dataset/198/steel+plates+faults) | 1,941 / 27 | UCI links no paper; not searched further | Which fault does a steel plate have? | Seven classes, some of them small |

## Ruled out: the creators did the same analysis

| # | Dataset | Creators' paper |
| --- | --- | --- |
| 1 | [Student Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) | Martins et al. (2021), early prediction of student performance |
| 2 | [Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance) | Cortez & Silva (2008), predicting secondary school performance |
| 4 | [Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) | Sakar et al. (2019), real-time prediction of purchasing intention |
| 5 | [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) | Moro, Cortez & Rita (2014), predicting telemarketing success |
| 6 | [Regensburg Pediatric Appendicitis](https://archive.ics.uci.edu/dataset/938/regensburg+pediatric+appendicitis) | Marcinkevičs et al. (2023), interpretable ML models for appendicitis |
| 7 | [Myocardial Infarction Complications](https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications) | Rossiev, Golovenkin et al. (1995), neural networks forecasting complications; follow-up work in 2021 |
| 11 | [Differentiated Thyroid Cancer Recurrence](https://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence) | Borzooei et al. (2023), ML risk stratification for recurrence |
| 12 | [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) | Yeh & Lien (2009), comparing data mining methods for default prediction |
| 13 | [Polish Companies Bankruptcy](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data) | Zięba, Tomczak & Tomczak (2016), boosted trees for bankruptcy prediction |
| 14 | [Phishing Websites](https://archive.ics.uci.edu/dataset/327/phishing+websites) | Mohammad, Thabtah & McCluskey (2012), assessing phishing website features |
| 16 | [Dry Bean](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) | Koklu & Özkan (2020), multiclass classification of dry beans |
| 18 | [Seoul Bike Sharing Demand](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand) | Sathishkumar & Cho (2020), three papers predicting this demand; UCI links none of them |
| 20 | [Superconductivity](https://archive.ics.uci.edu/dataset/464/superconductivty+data) | Hamidieh (2018), predicting critical temperature |

## Best options by goal

- **Cleanest fit with the requirement:** AIDS Clinical Trials (#10), because its creators did no machine learning at all.
- **Best for covering regression:** Communities and Crime (#19), because its target is a number and its many features suit Ridge and Lasso.
- **Largest dataset:** Diabetes 130-US Hospitals (#8), if the partial overlap with the creators' paper is acceptable.

## Next step

Choose between AIDS Clinical Trials, Communities and Crime and the non-UCI options in `datasets_other_sources.md`. Then check the chosen dataset for leakage, signal and baseline model performance, as was done for Student Dropout.
