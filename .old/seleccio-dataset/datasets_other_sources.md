# Project: candidate datasets from non-UCI sources

Fifteen tabular datasets from government open-data portals, surveys and open platforms. None are about housing. Most of these data publishers collect data for statistics or operations rather than to publish a machine learning model, so the project's question is less likely to repeat the creators' own paper.

**Recommendation: DGT accident microdata (#3).** It is official, recent, one row per accident, and can be filtered to Catalonia. The Catalan dataset (#2) is a good alternative if its download page checks out.

## How this list was checked

- **Checked (2026-11-09):** each dataset's existence, publisher and access route, through web searches. Sizes are only given where a source stated them.
- **Not checked:** none of the files has been downloaded or modelled. The "watch out for" notes come from the search results and general knowledge.
- **Creators' own machine learning work:** only Chicago food inspections (ruled out below) and StatsBomb were found to have it. The rest were not searched one by one, so this should be checked for the final choice.
- **Effort:** how much work it takes to get a modelling table. Low means a ready CSV, Medium means filtering or joining files, High means building the table from raw records.

## Road safety

| # | Dataset | What a row is | Question to answer | Effort | Watch out for |
| --- | --- | --- | --- | --- | --- |
| 1 | [Barcelona: accidents managed by the Guàrdia Urbana](https://opendata-ajuntament.barcelona.cat/data/en/dataset/accidents-tipus-gu-bcn) (Open Data BCN, CC BY 4.0) | An accident, in yearly CSVs from at least 2013 | Which accidents in Barcelona cause serious injuries? | Medium | Types, causes and people are in separate datasets that need joining. Confirm which dataset has the injury counts. |
| 2 | Catalonia: *Accidents de trànsit amb morts o ferits greus a Catalunya* (Dades Obertes, cited by [Idescat](https://idescat.cat/paae/2024/041003)) | An accident with deaths or serious injuries | Among serious accidents, which ones end in a death? | Low to Medium | The download page wasn't found, only references to it. Deaths are a small minority, so recall and PR curves matter. |
| 3 | [DGT accident microdata, Spain](https://www.dgt.es/menusecundario/dgt-en-cifras/dgt-en-cifras-resultados/dgt-en-cifras-detalle/Ficheros-de-microdatos-de-accidentes-con-victimas-2024/) (yearly files 2022-2025) | An accident with victims, urban or interurban | What makes an accident fatal or serious rather than minor? | Medium | Large and very imbalanced. Fields recorded after the accident (such as victim counts) can leak the outcome. |
| 4 | [UK road safety data (STATS19)](https://data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-accidents-safety-data) | A collision, with separate vehicle and casualty tables | Can road, weather and time conditions predict severity? | Medium | Three tables to join. Much third-party work exists, though not by the publisher. |

## Health surveys

| # | Dataset | What a row is | Question to answer | Effort | Watch out for |
| --- | --- | --- | --- | --- | --- |
| 5 | [CDC BRFSS 2024](https://cdc.gov/brfss/annual_data/annual_2024.html) | A survey respondent (457,670 records) | Which lifestyle and health factors predict diabetes or heart disease? | Medium | UCI's CDC Diabetes dataset comes from BRFSS 2015, so use a recent year and your own variable choice. Needs subsampling for SVM and k-NN. |
| 6 | [NHANES](https://wwwn.cdc.gov/nchs/nhanes/tutorials/datasets.aspx) | A participant, across questionnaire, exam and lab files | Can questionnaire and exam data predict hypertension or diabetes? | High | Files must be joined by participant ID. Lab values such as HbA1c partly define diabetes, so they leak the target. |

## Society and opinion surveys

| # | Dataset | What a row is | Question to answer | Effort | Watch out for |
| --- | --- | --- | --- | --- | --- |
| 7 | [Stack Overflow Developer Survey 2025](https://survey.stackoverflow.co/) (ODbL) | A developer (over 49,000 from 177 countries) | What predicts a developer's salary or their use of AI tools? | Medium | Salary gives a regression target. Many multi-choice answers need encoding. Respondents are self-selected. |
| 8 | [European Social Survey, Round 11](https://europeansocialsurvey.org/data-portal) | A respondent in one of 31 countries | Which social factors predict life satisfaction or trust in institutions? | Medium | Registration may be required. Survey weights should be considered. |
| 9 | [CEO Baròmetre d'Opinió Política](https://cran.nics.utk.edu/cran/web/packages/CEOdata/vignettes/using_CEOdata.html) (Centre d'Estudis d'Opinió) | A respondent in Catalonia, accumulated since 2014 | Which profiles predict vote intention or political attitudes? | Low to Medium | Politically sensitive, so frame the question carefully. Available as CSV/SAV files or through the `CEOdata` R package. |

## Education, transport and consumers

| # | Dataset | What a row is | Question to answer | Effort | Watch out for |
| --- | --- | --- | --- | --- | --- |
| 10 | [College Scorecard](https://collegescorecard.ed.gov/data/) (US Department of Education) | A US college or study programme | Which college characteristics predict graduates' earnings? | Medium | Regression target. Many values are missing or suppressed for privacy. |
| 11 | [BTS airline on-time performance](https://transportation.libanswers.com/faq/154783) | A US domestic flight | Which flights will arrive more than 15 minutes late? | Medium | Very large, downloaded one month at a time. Delay causes and departure delay are only known afterwards. |
| 12 | [NYC restaurant inspection results](https://catalog.data.gov/dataset/dohmh-new-york-city-restaurant-inspection-results) | A violation, repeated per inspection | Which restaurants will fail to get an A grade? | Medium | Rows must be grouped per inspection. The grade comes from the score, so exclude the score. |
| 13 | [Open Food Facts](https://wiki.openfoodfacts.org/Reusing_Open_Food_Facts_Data) (ODbL) | A food product | Can nutrient values alone predict whether a food is ultra-processed (NOVA 4)? | High | Very large tab-separated export with many gaps. Nutri-Score is a formula of the nutrients, so it's a poor target. The project runs its own ML tools, which weren't checked for NOVA. |

## Sport and games

| # | Dataset | What a row is | Question to answer | Effort | Watch out for |
| --- | --- | --- | --- | --- | --- |
| 14 | [StatsBomb open data](https://github.com/statsbomb/open-data) (non-commercial use) | A shot, extracted from match event files | What makes a shot become a goal (an expected-goals model)? | High | JSON files need converting into a shots table. StatsBomb's own xG value is in the data: exclude it as an input, but it makes a good benchmark. |
| 15 | [Lichess open database](https://database.lichess.org/) (CC0) | A rated online chess game | Can ratings, time control and opening predict who wins? | Medium | Monthly files are many GB. Reading only the game headers keeps it manageable. |

## Ruled out

- **Chicago food inspections:** the city's health department built and published its own model predicting critical violations, and an [independent audit](https://arxiv.org/abs/1910.04906) studied it.
- **OpenML:** most of its classic tabular datasets are copies of UCI or Kaggle datasets, so it doesn't add new options.

## Best options by goal

- **Most local and relevant:** Catalonia (#2) or Barcelona (#1) accidents, which are official data with no known model by the publisher.
- **Best for covering regression:** Stack Overflow salary (#7) or College Scorecard earnings (#10).
- **Best for discussing metrics:** DGT (#3) or Catalonia (#2) accidents, because fatal accidents are rare and missing them is costly.
- **Easiest start:** CEO Baròmetre (#9), which has ready-made microdata files.
- **Most original:** StatsBomb expected goals (#14), with a professional model to compare against.

## Next step

Shortlist two or three datasets, then check each for availability, leakage, signal and baseline model performance, as was done for the hair CSV (`preguntes_projecte_csv.md`) and Student Dropout (`student_dropout_summary.md`).
