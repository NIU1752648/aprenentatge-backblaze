# Context: processament de dades de Backblaze

Estat a 24-11-2026. L'estudi de viabilitat i els resultats són a `../estudi_backblaze.md`. Aquest fitxer recull com estan processades les dades i què queda pendent.

## Objectiu del projecte

- Dataset: Backblaze Drive Stats (fallades de discs durs), per a l'assignatura d'Aprenentatge Computacional.
- Requisits del professor:
  - Construir l'objectiu fixant un horitzó de temps t (el disc falla abans de t).
  - Entrenar diversos models per horitzó o per franja de temps per obtenir una probabilitat acumulada.
  - Centrar el treball en l'aplicació del model.
  - Els creadors del dataset no poden haver publicat la mateixa anàlisi.
- Enfocament: predir fallades a 7, 30 i 60 dies per fer manteniment predictiu.
- Reserva important: Backblaze decideix que un disc ha fallat en part pels comptadors SMART 5, 187, 188, 197 i 198, que el model també fa servir com a variables.

## Entorn

- Python: entorn gestionat amb uv des de l'arrel del projecte (`pyproject.toml`, `uv.lock`). Els scripts s'executen amb `uv run python -I scripts/<script>.py ...` des de `dataset/backblaze/`.
- Els scripts s'executen des de `backblaze/` i reben els camins com a arguments.

## Dades disponibles

| Fitxer | Contingut |
| --- | --- |
| `data/zip/data_<trimestre>.zip` | Zips originals del Q1 2025 al Q2 2026 (7,3 GB). Els CSV s'han esborrat després de consolidar |
| `data/panel/<trimestre>.parquet` | Sis panels (2025Q1 … 2026Q2), una fila per disc i dia, un grup de files per dia, zstd. 5,1 GB en total, 546 dies |
| `data/panel/<trimestre>_columns.csv` | Proporció d'absents de cada SMART `_raw`. Els sis trimestres tenen les mateixes 93 columnes i en conserven les mateixes 63 |
| `data/drives_q/<trimestre>.parquet` | Agregació per disc de cada trimestre (pas intermedi de `drives.parquet`) |
| `data/drives.parquet` | Una fila per disc: 372.456 discs, 6.861 fallades. Censura: `alive_at_end` 355.203, `left_data` 10.392 |
| `out/daily_stats.parquet`, `landmark_stats.parquet`, `failure_history.parquet`, `drive_onsets.parquet` | Sortides de `daily_stats.py` (sèries diàries, vegeu més avall) |
| `out/figures/*.png` | Figures del notebook `notebooks/01_dades.py` |
| `data/presence.parquet`, `data/presence_days.parquet` | Sortides de `presence.py`: un bit per disc i dia (1,4 MB) i, per dia, discs presents, esperats i caigudes generals |
| `out/landmarks/L=<data>.parquet` | Sortida del `build.py` nou: 65 dimecres (dev 40, gap 8, test 17), 21,4 M de files, 600 MB. Resum per data a `out/build_summary.csv` |
| `out/results_*.csv`, `model_log.txt`, `perm_importance_rf_t30.csv`, `cumulative_vs_km.csv` | Resultats de l'estudi amb el Q4 2025 sol |

## Flux de processament

1. **`scripts/consolidate.py <dir csv> <dir sortida> <etiqueta>`**, per exemple `consolidate.py data/q4 data 2025Q4`. Triga uns 200 s per trimestre.
   - Primer pas: llegeix els CSV dia a dia amb un esquema comú. Inclou la unió de tots els SMART `_raw` (93 columnes) i compta els valors absents.
   - Segon pas: conserva les columnes amb menys d'un 99,9 % de valors absents (63 de 93) i codifica `model`, `datacenter` i `is_legacy_format` com a diccionari.
   - Desa l'agregació per disc del trimestre a `drives_q/<etiqueta>.parquet` i reconstrueix `drives.parquet` combinant **totes** les agregacions. Així mai no hi ha més d'un trimestre a memòria, i per afegir un trimestre només cal consolidar-lo.
   - Els lectors del panel han d'unificar els esquemes dels trimestres (`pa.unify_schemas`), com fa `build.py`, per si algun trimestre conservés columnes diferents.
   - Triga uns 4 min per trimestre en aquesta màquina (Windows). Els CSV de cada trimestre es descomprimeixen en un directori nou `data/csv_<etiqueta>/` i s'esborren després.
   - Columnes de `drives.parquet`:
     - Identificació: `serial_number`, `model`, `capacity_bytes`, `datacenter`.
     - Període observat: `first_seen`, `last_seen`, `n_days`.
     - Fallada: `fail_date` (primera fallada en tot el període), `failed`, `end_date`, `duration_days`.
     - Presència: `absent_days` (dies sense fila entre `first_seen` i `last_seen`) i `n_quarters`.
     - `censoring`: `failed` (945 discs), `alive_at_end` (338.815) o `left_data` (1.904).
     - Hores d'ús (SMART 9): `power_on_hours_first` i `power_on_hours_last`.
2. **`scripts/presence.py <dir data>`**, per exemple `presence.py data`. Uns 3 min. Llegeix només `date` i `serial_number` dia a dia i escriu:
   - `data/presence.parquet`: per disc (mateix ordre que `drives.parquet`), `bits` de 69 bytes amb `np.packbits(..., bitorder="little")`; el bit i diu si el disc té fila el dia i (0 = 1-1-2025).
   - `data/presence_days.parquet`: per dia, `n_present`, `n_expected` (entre `first_seen` i `last_seen` o la fallada), `absent_share` i `outage` (> 5 %).
   - Comprovat: els bits quadren amb `n_days`, `first_seen` i `last_seen` de tots els discs. Només hi ha una caiguda general, el 12-5-2025 (22,3 % dels esperats; el 29 % que es deia abans comparava amb els presents). La resta de dies no passen del 0,8 %, així que el llindar del 5 % no és delicat.
3. **`scripts/build.py <dir data> <dir sortida>`**, per exemple `build.py data out/landmarks`. Uns 4 min, una sola passada pel panel. Reescrit el 22-11-2026 (tasca #32).
   - Dates `L`: tots els dimecres del 5-2-2025 al 29-4-2026, un fitxer per data, amb `split` = `dev` (fins al 5-11-2025, 40), `gap` (8) o `test` (des del 7-1-2026, 17).
   - Població (decisió 19, 24-11-2026): discs vistos dins de [L−7, L] que no han fallat a L o abans, amb els valors de la seva última fila; `days_since_seen` = dies des d'aquesta fila. S'exclouen els discs absents a L que no tornen mai sense fallar (etiquetes desconegudes a tots els horitzons). `model` i `capacity_bytes` surten de `drives.parquet`.
   - Variables: G0 (`model`, `manufacturer`, `capacity_bytes`, `age_years`), G1 (SMART 1, 7, 10, 12, 183, 184, 189, 190, 192, 193, 194, 199, 240, 241, 242 i `d30_241`, `d30_242`), G2 (SMART 5, 187, 188, 197, 198 crus), G3 (`d1_`, `d7_`, `d30_` de cada comptador d'error) i GO (`absent_30` sense dies de caiguda general, `days_since_return`, `days_since_first`, `seen_at_start`, `days_since_seen`).
   - Canvis: últim valor a L menys l'última fila del disc dins de [L−k−7, L−k]; NaN si no n'hi ha. A la primera data la finestra de 30 dies queda retallada a l'1-1-2025.
   - Etiquetes: `event`, `t_obs` (fins a la fallada o fins a `last_seen` de tot el període) i `y_7`, `y_30`, `y_60` (NaN si el disc surt abans de L+t sense fallar). Les franges es deriven de `t_obs` i `event`.
   - No aplica la censura de festes, ni mostreig ni imputació: es fan en llegir (tasca #34).
   - SMART 1, 7, 10, 188, 192, 193, 241 i 242 es guarden en float64 perquè passen de 2^24.
4. **`scripts/daily_stats.py <dir data> <dir sortida>`**, per exemple `daily_stats.py data out`. Uns 2,5 min. Recorre el panel dia a dia una sola vegada amb estat vectoritzat per disc (màscara de bits de presència de 64 dies, últim valor de la suma dels comptadors d'error) i escriu:
   - `daily_stats.parquet`: per dia, discs actius, primeres fallades, discs nous, que tornen i que surten sense fallar, discs amb algun comptador d'error > 0, amb pujada i amb primera pujada.
   - `landmark_stats.parquet`: per a cada dimecres, proporció de discs amb dies absents i amb pujades de comptadors als 7 i 30 dies anteriors.
   - `failure_history.parquet`: per a cada disc que falla, dies absents als 7 i 30 dies anteriors, dies des de l'observació anterior, comptadors a la fallada, data de la primera pujada i de l'última.
   - `drive_onsets.parquet`: data de la primera pujada observada (de 0 a > 0) de cada disc.
5. **`scripts/explore.py <landmarks>`** (format antic d'un sol fitxer; cal adaptar-lo): recomptes per horitzó, regla de Backblaze (algun comptador d'error > 0) i valors absents per model.
6. **`scripts/model.py <landmarks> <dir sortida>`** (format antic; cal adaptar-lo):
   - Models: regressió logística, k-NN, Naive Bayes, arbre, random forest i SVM.
   - Validació: `GroupKFold` per disc (3 grups).
   - Entrenament: totes les fallades i una mostra de discs sense fallada, amb les probabilitats corregides per la mida de la mostra.
   - Transformacions: `log1p` als SMART i logaritme amb signe als `d7`.

**Atenció:** amb sis trimestres, el Q4 2025 aporta 932 fallades noves en lloc de 945: 13 discs ja havien fallat en un trimestre anterior i ara només es compta la primera fallada. Els recomptes de l'estudi (323, 1.422 i 2.331 a 7, 30 i 60 dies) es van fer amb el Q4 sol i no coincidiran exactament amb la taula nova.

## Anàlisi de Nadal (fet el 20-11-2026)

Notebook `notebooks/01_dades.py`, amb dades fins al 4-1-2026 (regla del pla, secció 1.5 bis). Resum:

- **La caiguda comença la setmana de Thanksgiving (17-11), no per Nadal.** Del 17-11-2025 al 4-1-2026, 1,80 fallades per 100.000 discs-dia contra 4,14 les set setmanes anteriors (−57 %). Les pujades de comptadors d'error augmenten un 10 % i les primeres pujades no canvien. Els discs es deterioren igual i se'n declaren menys com a fallats.
- **És retard en la retirada** (matisat el 25-11: les fallades no es recuperen després; vegeu la revisió de sota): al desembre només un 1,0 % dels discs amb primera pujada d'errors consten com a fallats en 14 dies (1,8–2,2 % de setembre a novembre). Les fallades del desembre són més evidents: més comptadors > 0 i més absències prèvies.
- **No hi ha acumulació clara al gener de 2025** (+6 % i +10 % les dues primeres setmanes senceres).
- **Etiquetes de desenvolupament afectades:** a 7 dies cap, a 30 dies 3 dates (unes 150 fallades que falten), a 60 dies 7 dates (unes 1.000, fins a un 45 % de la taxa a la data del 5-11).
- **Dia de la setmana:** les fallades es registren un 35 % menys en diumenge i dilluns, i les sortides sense fallar es concentren en dilluns. Les dates setmanals sempre en dimecres neutralitzen l'efecte.
- **Caiguda de reporting el 12-5-2025:** 70.446 discs absents aquell dia (22 % dels esperats). Infla els dies d'absència de les dates de maig i juny.

**Revisió amb tot el període (25-11-2026).** Resposta del professor a la pregunta 2: el submostreig està bé (no és el seu estil, però dona millors resultats). La caiguda de festes només justifica tractar-les a part si després el ritme torna a pujar. Ara `01_dades.py` mostra les dades fins al 30-6-2026, incloent-hi el test (només taxes descriptives), amb una figura nova, `out/figures/nadal_balanc.png`:

- **El ritme torna a pujar, i la decisió i el split es mantenen.** Les pujades de comptadors d'error no baixen mai. Les fallades registrades tornen a pujar just després de Reis: 2,8 per 100.000 discs-dia la primera setmana de gener, 3,2–3,3 al gener–febrer i 4,55 (nivell normal) del març al juny, contra 4,14 abans de festes.
- **Les fallades de festes no es recuperen més tard** (unes 390 que falten, sense cap excés al gener). Per això es diu «durant les festes es registren menys fallades», no «es retiren més tard». La pujada de l'abril–maig de 2026 és de models antics i no s'atribueix a les festes.
- **Primeres dates de test:** les etiquetes de gener miren cap al gener–febrer, que té un 20 % menys de fallades registrades. Es manté la prova de sensibilitat de treure les dues primeres dates de test, i les mètriques es donaran també per data.
- La setmana del 5 al 11-1-2026 surten 4.640 discs sense fallar (unes sis vegades el nivell de la tardor). Hipòtesi per comprovar: discs deteriorats no retirats per festes que surten sense constar com a fallada.

**Mesura de l'error d'etiqueta** (`notebooks/01b_etiquetes.py`, 20-11-2026). Mostra de les 40 dates de desenvolupament amb tots els discs que fallen i un 5 % de la resta, amb pesos; es desa a `out/label_study.parquet`. Models GB i logística, validació creuada per disc, t = 30 i 60:

- **Dependència de la regla humana:**
  - La regla del tècnic (comptadors que pugen) ja captura un 46 % de les fallades a 30 dies amb l'1 % d'alertes. El GB en captura un 52 %.
  - Tots dos gairebé no troben les fallades netes (sense errors; un 24 % de les positives): 0,7 % i 3 %. El GB sense els 5 comptadors en troba un 27 %.
- **Falsos positius:** un 9 % fallen entre 30 i 90 dies després (contra un 0,15 % dels TN) i un 96 % ja tenen errors. Comptant-los com a encerts, la precisió passa d'un 6 % a un 14 %.
- **Festes, t = 60:**
  - Entrenar amb les etiquetes contaminades no canvia el rànquing (B − A en recall: −0,002 [−0,006, 0,001]), però rebaixa un 4 % les probabilitats.
  - El model recalibrat estima que falten unes 1.950 fallades-etiqueta a les 7 dates afectades (32 %). El control a les dates netes dona +2 %.
- **Soroll simulat:** amagant fins a la meitat de les fallades, el recall@1 % no es mou (0,46). La calibració relativa cau com 1 − p.
- **Sobreajust per disc:** amb fulles de 200 files, el GB preveia 2,2 vegades les fallades en discs nous. Cal regularitzar fort (fulles de 2.000 files) i, tot i així, queda un excés del 25–30 % amb una mostra del 5 %. Per a `build.py`, cal pensar en una mostra de negatius més gran o recalibrar.

**Opció triada** (decidida el 24-11-2026, Vikunja #70):

- Censura administrativa al **16-11-2025** per a les etiquetes de desenvolupament. Per a l'horitzó t només entren les dates amb L + t ≤ 16-11-2025, i al model per franges es censuren les franges posteriors.
- Sense variable de calendari.
- El test comença el 7-1-2026, amb una comprovació de sensibilitat sense les dues primeres dates.
- Dies d'absència sense comptar els dies de caiguda general (més del 5 % del parc absent).
- **Cap dada posterior al tall a l'entrenament.** Les etiquetes de desenvolupament es recalculen en llegir amb la informació fins al 16-11-2025: fallada si `fail_date` ≤ tall, i viu fins a L + t només si el disc té fila a L + t o després **i abans del tall** (presència), no amb el `last_seen` de tot el període que porta `t_obs`. Les llistes de models freqüents, medianes i calibratge, només amb el desenvolupament.

## Resultats del `build.py` nou

**Versió del 24-11-2026 (població dels 7 dies anteriors), la vigent.** 5,5 min.

| Bloc | Dates | Files | Positius y_7 / y_30 / y_60 | Absents el dia L |
| --- | --- | --- | --- | --- |
| dev | 40 | 12.837.652 | 3.438 / 15.075 / 28.977 | 8.081 |
| gap | 8 | 2.706.352 | 347 / 1.520 / 3.693 | 5.058 |
| test | 17 | 5.826.781 | 1.632 / 7.277 / 14.516 | 10.320 |

- Ara 5.417 de les 5.714 primeres fallades (94,8 %) cauen a la franja (0, 7] d'un dimecres, contra 4.862 (85 %) abans. Les 297 que falten són discs absents més de 7 dies abans de fallar o que apareixen després del dimecres.
- Discs exclosos per marxar sense tornar ni fallar: 4.495 files a dev i 4.369 a test.

**Versió del 22-11-2026 (només discs amb fila el dia L), substituïda:**

| Bloc | Dates | Files | Positius y_7 / y_30 / y_60 | Discs positius a 60 dies |
| --- | --- | --- | --- | --- |
| dev | 40 | 12.829.571 | 3.152 / 14.426 / 28.178 | 3.867 |
| gap | 8 | 2.701.294 | 311 / 1.427 / 3.547 | — |
| test | 17 | 5.816.461 | 1.399 / 6.509 / 13.374 | 2.359 |

- Etiquetes desconegudes: menys d'un 0,7 % a cada horitzó. Canvis a 30 dies amb NaN: un 1,9 % (gairebé tots discs vistos fa menys de 30 dies).
- El 98,9 % de les files tenen `absent_30` = 0. A dev, la taxa de y_30 és del 0,96 % amb absències i del 0,107 % sense (unes 9 vegades).
- El 12-5-2025 no infla `absent_30` a les dates de maig.
- Es veu la caiguda de festes: y_60 baixa de ~800 per data a l'estiu a 403 el 5-11-2025.

**Per què es va canviar la població (decisió 19).** De les 5.714 primeres fallades entre el 5-2-2025 i el 6-5-2026, 852 (15 %) no cauen a la franja (0, 7] de cap dimecres: 832 són discs que no tenien fila el dimecres anterior (forat mitjà de 5 dies, un quart tornen el dia abans de fallar) i 20 apareixen per primera vegada després. Són precisament els discs que GO vol captar. Es va decidir el 24-11-2026 fer servir els discs vistos dins de [L−7, L].

## Validació creuada al homelab (scripts del 24-11-2026)

Dos scripts que s'executen al homelab, en aquest ordre, des de `dataset/backblaze/`. Necessiten `out/landmarks/`, `data/drives.parquet` i `data/presence.parquet`.

1. **`scripts/learning_curve.py out/landmarks data out/lc --jobs N [--max-mem-gb 12]`** (decisió 9). Corba d'aprenentatge amb hiperparàmetres fixos (gradient boosting, gradient boosting amb fulla escalada i logística), per a les franges i y_30. Eixos: fracció de discs negatius {1, 2, 5, 15, 30, 50, 100} % amb totes les seves dates, i aprimament de dates {1, 1/3, 1/10} igual per a positius i negatius. Uns 5–8 h, 756 entrenaments; el gradient boosting amb totes les dades arriba a 9–10 GB i la logística a 14 GB (per defecte se salta el que passa de 12 GB). Resultat: `out/lc/lc_recommendation.json` amb la mida més petita a una desviació estàndard de la millor.
2. **`scripts/cv_models.py out/landmarks data out/cv --jobs N [--neg-drive-frac F] [--date-step 1]`** (E3). Cerca d'hiperparàmetres de gradient boosting i logística amb `GroupKFold` per disc (K = 3, per hash), horitzons separats i franges, regla d'una desviació estàndard. Per defecte entrena amb el 15 % dels discs negatius (mida triada amb la corba d'aprenentatge, 25-11-2026). Unes 8–14 h: la part lenta és la logística L1 (`saga`, un sol fil). Amb `--models gb` i després `--models lr` al mateix directori es pot tenir el GB primer. Exposa les funcions comunes (etiquetes estrictes, particions, mostreig, mètriques) que també fa servir `learning_curve.py`.

Tots dos només llegeixen les 40 dates de desenvolupament, tornen a calcular les etiquetes amb el tall estricte del 16-11-2025, desen cada resultat de seguida (es poden reprendre) i registren el calibratge mitjà (probabilitat corregida / taxa observada) per detectar errors de mostreig. Els CSV, JSON, PNG i logs de `out/lc/` i `out/cv/` es versionen; els parquet de prediccions, no.

**Etiquetes estrictes (comprovat):** respecte de les desades, només canvien 338, 434 i 633 files (y_7, y_30, y_60), totes de 0 a desconegut. Positius al desenvolupament: y_7 3.438, y_30 14.038, y_60 24.569.

**Avís sobre el mostreig de negatius:** no s'han d'aprimar les dates dels negatius amb un subconjunt global mentre es guarden els positius de totes les dates. `days_since_first` amb `seen_at_start` identifica la data, i el model aprèn «data fora de la mostra de negatius ⇒ positiu» (en una prova, probabilitats unes 180 vegades massa altes). Es mostreja per disc amb totes les dates; si s'aprimen dates, ha de ser igual per a les dues classes o amb una fase diferent per disc.

## Decisió en curs: separació entre entrenament i test

Proposta discutida, encara no implementada:

- **Ajuntar diversos trimestres** (per exemple, del primer al quart de 2025, més el primer de 2026 per completar les etiquetes). Així es tenen més fallades (unes 4.000 l'any en lloc de 945), horitzons de 60 dies per a totes les dates de referència i la possibilitat d'estudiar els canvis en el parc de discs.
- **Test separat en el temps:** dates de referència d'un període final (per exemple, novembre–desembre de 2025) que no es fan servir fins a l'avaluació final.
- **Separació de 60 dies** entre l'última data d'entrenament i la primera de test, perquè les etiquetes d'entrenament no vegin fallades del període de test.
- **k-fold només dins del període de desenvolupament**, amb `GroupKFold` per disc, per triar models i hiperparàmetres.
- Es pot donar un segon resultat amb només els discs nous al test, per a una avaluació més estricta.
- **Inferència:** bootstrap per disc sobre el test per obtenir intervals de confiança de les diferències de PR-AUC i del percentatge detectat amb l'1 % d'alertes entre models.
- Considerar dates de referència cada 2 setmanes per reduir la correlació entre files.
- Cal tractar els models de disc que no apareguin a l'entrenament (una categoria "desconegut" o agrupar per fabricant).

## Pendents

- [x] Descarregar i consolidar els sis trimestres (Q1 2025 – Q2 2026). Fet el 20-11-2026.
- [x] `consolidate.py`: taula de discs agregada per trimestre i esquema unificat a `build.py`.
- [x] Anàlisi preliminar de Nadal (1.5 bis del pla).
- [x] Decisió de Nadal confirmada (censura al 16-11-2025). 24-11-2026.
- [x] Taula de presència (`presence.py`). Fet el 22-11-2026.
- [x] `build.py` nou amb dates setmanals, `split`, grups G0–G3 i GO, canvis a 1, 7 i 30 dies. Fet el 22-11-2026.
- [x] Població a L: discs vistos en els 7 dies anteriors; `out/landmarks` regenerat el 24-11-2026.
- [ ] Validació creuada de gradient boosting i logística al homelab (`cv_models.py`) i corba d'aprenentatge per a la decisió 9.
- [ ] Partició i mostra d'entrenament (#34): censura de festes en llegir, negatius per hash de disc, pesos.
- [ ] Adaptar `explore.py` i `model.py` al format nou (un fitxer per data).
- [ ] Es poden esborrar els zips de `data/zip/` (7,3 GB) si cal espai: es tornen a baixar en uns 6 minuts.
- [ ] Ajustar els hiperparàmetres amb validació creuada dins del període de desenvolupament.
- [ ] Calibrar les probabilitats (Platt o regressió isotònica).
- [ ] Provar un model per franges de temps per evitar les corbes acumulades que baixen.
- [ ] Fer el bootstrap per disc per comparar models al test.
