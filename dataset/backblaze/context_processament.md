# Context: processament de dades de Backblaze

Estat a 12-11-2026. L'estudi de viabilitat i els resultats són a `../estudi_backblaze.md`. Aquest fitxer recull com estan processades les dades i què queda pendent.

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
| `data/data_Q4_2025.zip` | Zip original, 1,16 GB |
| `data/q4/` | CSV diaris de l'1-10-2025 al 31-12-2025 (92 fitxers, 12,2 GB) |
| `data/panel/2025Q4.parquet` | Una fila per disc i dia: 30.941.708 files, 889 MB, un grup de files per dia, zstd |
| `data/panel/2025Q4_columns.csv` | Proporció de valors absents de cada SMART `_raw` i si s'ha conservat |
| `data/drives.parquet` | Una fila per disc (341.664), es reconstrueix a partir de tots els trimestres de `panel/` |
| `out/landmarks.parquet` | Taula de dates de referència: 1.337.354 files, 335.996 discs |
| `out/results_*.csv`, `model_log.txt`, `perm_importance_rf_t30.csv`, `cumulative_vs_km.csv` | Resultats dels models (vegeu l'estudi) |

## Flux de processament

1. **`scripts/consolidate.py <dir csv> <dir sortida> <etiqueta>`**, per exemple `consolidate.py data/q4 data 2025Q4`. Triga uns 200 s per trimestre.
   - Primer pas: llegeix els CSV dia a dia amb un esquema comú. Inclou la unió de tots els SMART `_raw` (93 columnes) i compta els valors absents.
   - Segon pas: conserva les columnes amb menys d'un 99,9 % de valors absents (63 de 93) i codifica `model`, `datacenter` i `is_legacy_format` com a diccionari.
   - Reconstrueix `drives.parquet` a partir de **tots** els fitxers de `panel/`, així que per afegir un trimestre només cal consolidar-lo.
   - Columnes de `drives.parquet`:
     - Identificació: `serial_number`, `model`, `capacity_bytes`, `datacenter`.
     - Període observat: `first_seen`, `last_seen`, `n_days`.
     - Fallada: `fail_date`, `failed`, `end_date`, `duration_days`.
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
3. **`scripts/explore.py <landmarks>`**: recomptes per horitzó, regla de Backblaze (algun comptador d'error > 0) i valors absents per model.
4. **`scripts/model.py <landmarks> <dir sortida>`**:
   - Models: regressió logística, k-NN, Naive Bayes, arbre, random forest i SVM.
   - Validació: `GroupKFold` per disc (3 grups).
   - Entrenament: totes les fallades i una mostra de discs sense fallada, amb les probabilitats corregides per la mida de la mostra.
   - Transformacions: `log1p` als SMART i logaritme amb signe als `d7`.

**Atenció:** `out/landmarks.parquet` (9-11, 12:05) és anterior a la versió actual de `build.py` (9-11, 18:07). Aquesta versió ja llegeix el panel consolidat en lloc dels CSV, però encara no s'ha tornat a executar. Abans de seguir, convé regenerar la taula i comprovar que els recomptes coincideixen: 323, 1.422 i 2.331 fallades a 7, 30 i 60 dies.

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

- [ ] Regenerar `out/landmarks.parquet` amb el `build.py` actual i verificar els recomptes.
- [ ] Descarregar i consolidar més trimestres, cadascun en un directori nou (per exemple `data/q1_2025/`): `consolidate.py data/<dir> data <etiqueta>`.
- [ ] Afegir a `build.py` la partició desenvolupament / separació / test i fer configurable la freqüència de les dates de referència.
- [ ] Ajustar els hiperparàmetres amb validació creuada dins del període de desenvolupament.
- [ ] Calibrar les probabilitats (Platt o regressió isotònica).
- [ ] Provar un model per franges de temps per evitar les corbes acumulades que baixen.
- [ ] Fer el bootstrap per disc per comparar models al test.
