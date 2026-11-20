# Context: processament de dades de Backblaze

Estat a 20-11-2026. L'estudi de viabilitat i els resultats són a `../estudi_backblaze.md`. Aquest fitxer recull com estan processades les dades i què queda pendent.

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
| `out/landmarks.parquet` | **No existeix en aquesta màquina.** Cal regenerar-lo amb el `build.py` nou |
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
2. **`scripts/build.py <dir data> <parquet sortida>`**, per exemple `build.py data out/landmarks.parquet`.
   - Dates de referència `L`: setmanals, el mateix dia de la setmana que el primer dia. Cal que existeixi `L - 7` i que `L + 60` no passi de l'últim dia. Amb el quart trimestre surten el 8, 15, 22 i 29 d'octubre.
   - Variables:
     - Comptadors d'error: SMART 5, 187, 188, 197 i 198.
     - Altres SMART: 1, 7, 9, 10, 12, 183, 184, 189, 190, 192, 193, 194, 199, 240, 241 i 242.
     - `d7_<i>`: canvi de cada comptador d'error en els 7 dies anteriors.
   - Etiquetes: `y_t` = 1 si el disc falla dins de (L, L+t]. Val NaN si el disc surt de les dades abans de L+t sense fallar. També es guarden `t_obs` i `event` per a Kaplan-Meier.
   - Exclou els discs que ja han fallat a `L` o abans.
3. **`scripts/daily_stats.py <dir data> <dir sortida>`**, per exemple `daily_stats.py data out`. Uns 2,5 min. Recorre el panel dia a dia una sola vegada amb estat vectoritzat per disc (màscara de bits de presència de 64 dies, últim valor de la suma dels comptadors d'error) i escriu:
   - `daily_stats.parquet`: per dia, discs actius, primeres fallades, discs nous, que tornen i que surten sense fallar, discs amb algun comptador d'error > 0, amb pujada i amb primera pujada.
   - `landmark_stats.parquet`: per a cada dimecres, proporció de discs amb dies absents i amb pujades de comptadors als 7 i 30 dies anteriors.
   - `failure_history.parquet`: per a cada disc que falla, dies absents als 7 i 30 dies anteriors, dies des de l'observació anterior, comptadors a la fallada, data de la primera pujada i de l'última.
   - `drive_onsets.parquet`: data de la primera pujada observada (de 0 a > 0) de cada disc.
4. **`scripts/explore.py <landmarks>`**: recomptes per horitzó, regla de Backblaze (algun comptador d'error > 0) i valors absents per model.
5. **`scripts/model.py <landmarks> <dir sortida>`**:
   - Models: regressió logística, k-NN, Naive Bayes, arbre, random forest i SVM.
   - Validació: `GroupKFold` per disc (3 grups).
   - Entrenament: totes les fallades i una mostra de discs sense fallada, amb les probabilitats corregides per la mida de la mostra.
   - Transformacions: `log1p` als SMART i logaritme amb signe als `d7`.

**Atenció:** amb sis trimestres, el Q4 2025 aporta 932 fallades noves en lloc de 945: 13 discs ja havien fallat en un trimestre anterior i ara només es compta la primera fallada. Els recomptes de l'estudi (323, 1.422 i 2.331 a 7, 30 i 60 dies) es van fer amb el Q4 sol i no coincidiran exactament amb la taula nova.

## Anàlisi de Nadal (fet el 20-11-2026)

Notebook `notebooks/01_dades.py`, amb dades fins al 4-1-2026 (regla del pla, secció 1.5 bis). Resum:

- **La caiguda comença la setmana de Thanksgiving (17-11), no per Nadal.** Del 17-11-2025 al 4-1-2026, 1,80 fallades per 100.000 discs-dia contra 4,14 les set setmanes anteriors (−57 %). Les pujades de comptadors d'error augmenten un 10 % i les primeres pujades no canvien. Els discs es deterioren igual i se'n declaren menys com a fallats.
- **És retard en la retirada:** al desembre només un 1,0 % dels discs amb primera pujada d'errors consten com a fallats en 14 dies (1,8–2,2 % de setembre a novembre). Les fallades del desembre són més evidents: més comptadors > 0 i més absències prèvies.
- **No hi ha acumulació clara al gener de 2025** (+6 % i +10 % les dues primeres setmanes senceres).
- **Etiquetes de desenvolupament afectades:** a 7 dies cap, a 30 dies 3 dates (unes 150 fallades que falten), a 60 dies 7 dates (unes 1.000, fins a un 45 % de la taxa a la data del 5-11).
- **Dia de la setmana:** les fallades es registren un 35 % menys en diumenge i dilluns, i les sortides sense fallar es concentren en dilluns. Les dates setmanals sempre en dimecres neutralitzen l'efecte.
- **Caiguda de reporting el 12-5-2025:** 70.446 discs (29 %) absents aquell dia. Infla els dies d'absència de les dates de maig i juny.

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

**Opció triada** (pendent de confirmar per la parella):

- Censura administrativa al **16-11-2025** per a les etiquetes de desenvolupament. Per a l'horitzó t només entren les dates amb L + t ≤ 16-11-2025, i al model per franges es censuren les franges posteriors.
- Sense variable de calendari.
- El test comença el 7-1-2026, amb una comprovació de sensibilitat sense les dues primeres dates.
- Dies d'absència sense comptar els dies de caiguda general (més del 5 % del parc absent).

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
- [ ] Confirmar a Vikunja la decisió de Nadal (censura al 16-11-2025).
- [ ] `build.py` nou: partició desenvolupament / separació / test, censura de festes, dates setmanals, dies d'absència (sense caigudes generals), canvis a 7 i 30 dies. Regenerar `out/landmarks`.
- [ ] Es poden esborrar els zips de `data/zip/` (7,3 GB) si cal espai: es tornen a baixar en uns 6 minuts.
- [ ] Ajustar els hiperparàmetres amb validació creuada dins del període de desenvolupament.
- [ ] Calibrar les probabilitats (Platt o regressió isotònica).
- [ ] Provar un model per franges de temps per evitar les corbes acumulades que baixen.
- [ ] Fer el bootstrap per disc per comparar models al test.
