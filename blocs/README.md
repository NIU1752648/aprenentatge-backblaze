# Blocs de feina que queden (estat a 24-11-2026)

Cada document explica un bloc amb paraules senzilles: què és, per què el fem, com es fa, qui el fa, què necessita i quan el donem per acabat. El context general és a [`../posada_al_dia.md`](../posada_al_dia.md), i els detalls tècnics, a `../pla_projecte.md`.

| Bloc | Document | Qui | Quan |
| --- | --- | --- | --- |
| 1 | [Quantes dades fem servir per entrenar](01_mida_mostra.md) | Nico | dc. 25-11 |
| 2 | [Hiperparàmetres de gradient boosting i logística](02_hiperparametres.md) | Nico | 25–27-11 |
| 3 | [Els altres models](03_altres_models.md) | Nico (random forest, k-NN, SVM), Sergi (Naive Bayes) | 26–29-11 |
| 4 | [Referències, calibratge i bootstrap](04_referencies_calibratge_bootstrap.md) | Sergi (referències), Nico (calibratge i bootstrap) | 26–29-11 |
| — | **D2: models congelats** | tots dos | **dg. 29-11** |
| 5 | [Experiment central: de què ens podem refiar](05_experiment_central.md) | Nico (E1, E2, SHAP), Sergi (arbre dibuixat) | 30-11 – 3-12 |
| 6 | [Execució única del test](06_test.md) | Nico | **dv. 4-12** |
| 7 | [Aplicació: quins discs i quan, i esperança](07_aplicacio.md) | Nico (E5), Sergi (Kaplan-Meier) | 4–6-12 |
| 8 | [Notebooks i figures finals](08_notebooks.md) | Nico, amb figures de Sergi | 2–7-12 |
| 9 | [Presentació, teoria, assajos i entrega](09_presentacio.md) | Sergi (diapositives, teoria), tots dos | 7–13-12 |
| 10 | [Cerca bibliogràfica i professor](10_bibliografia_professor.md) | Sergi | aquesta setmana |

## Com treballem sense estar junts

- **El codi i els resultats petits passen pel git.** Els CSV, JSON i PNG de `out/` es versionen; cadascú fa `git pull` abans de començar i commit quan acaba una peça.
- **Les dades grans es passen un sol cop:** `out/landmarks/`, `data/drives.parquet` i `data/presence.parquet`.
- **Vikunja** diu qui fa cada tasca. Quan acabis una tasca, marca-la feta i escriu-hi un comentari curt amb el resultat.
- **Punts de trobada en línia:** dg. 29-11 (models congelats), dv. 4-12 (test), dc. 9-12 i dv. 11-12 (assajos).
- **Regla d'or:** cadascú ha de poder explicar la part de l'altre a la presentació, perquè les preguntes poden anar a qualsevol dels dos.
