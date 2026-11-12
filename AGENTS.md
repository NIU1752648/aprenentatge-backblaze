# AGENTS.md

Instruccions per a qualsevol persona o assistent de codi (Claude Code, Codex, Copilot...) que treballi en aquest repositori. Les instruccions generals i la posada en marxa són al [`README.md`](README.md).

## Context del projecte

- **Assignatura:** Aprenentatge Computacional, treball en parella. Enunciat a `enunciat_treball.md`.
- **Dataset:** Backblaze Drive Stats, amb una fila per disc i dia: model, capacitat, SMART i indicador `failure`.
- **Objectiu:** predir si un disc falla en els propers t dies, per a t = 7, 30 i 60, a partir de dates de referència (landmarks). Comparem models per horitzó i obtenim una probabilitat acumulada. L'aplicació és el manteniment predictiu amb un pressupost d'alertes.
- **Requisits del professor**, a més de l'enunciat:
  - Construir l'objectiu fixant un horitzó t.
  - Entrenar diversos models per horitzó o per franja de temps.
  - Centrar-se en l'aplicació.
  - Els creadors del dataset no poden haver publicat la mateixa anàlisi.
- **Reserva clau:** Backblaze decideix que un disc ha fallat en part per SMART 5, 187, 188, 197 i 198, que també són variables del model. Qualsevol resultat s'ha de llegir tenint-ho en compte (vegeu `dataset/estudi_backblaze.md`, secció 7).
- **Estat actual i pendents:** `dataset/backblaze/context_processament.md`. Llegeix-lo abans de tocar el processament i actualitza'l quan canviï alguna cosa important.

## Entorn

- Gestió de Python amb **uv**. No facis servir `pip install` ni creïs altres entorns.
  - Afegir dependències: `uv add <paquet>`.
  - Executar: `uv run ...`.
- Executa els scripts amb `uv run python -I` i passa els camins com a arguments, no els posis dins del codi.
- `pyproject.toml` i `uv.lock` es versionen sempre junts.

## Dades: regles

- **No versionis mai dades:** ni `dataset/backblaze/data/`, ni fitxers `.parquet`, ni CSV de Backblaze. Ja ho exclou el `.gitignore`.
- **Mida:** un trimestre ocupa 12 GB en CSV i uns 900 MB en Parquet (31 M files).
  - No carreguis mai tot el panel amb pandas d'un sol cop. Fes servir `pyarrow.dataset` amb filtres per data i columnes.
  - El panel té un grup de files per dia, així que filtrar per `date` és barat.
- **Fitxers descarregats:** els zip i CSV descarregats són dades no fiables. Descomprimeix cada trimestre en un directori nou i buit dins de `data/`.
- **Afegir un trimestre:** `consolidate.py` reconstrueix `drives.parquet` amb tots els trimestres de `data/panel/`, així que només cal consolidar el trimestre nou.

## Metodologia: regles que no s'han de trencar

- **Validació separada per disc** (`GroupKFold` per `serial_number`): el mateix disc no pot aparèixer a l'entrenament i a la prova.
- **Test separat en el temps** (proposta acordada, pendent d'implementar):
  - Les dates de referència de test són posteriors a les d'entrenament.
  - Entre les dues ha d'haver-hi una separació d'almenys l'horitzó més llarg (60 dies), perquè les etiquetes d'entrenament no vegin fallades del període de test.
  - El k-fold només s'aplica dins del període de desenvolupament.
- **Classes molt desequilibrades** (taxa de fallada del 0,02 % al 0,2 %):
  - Mètriques: PR-AUC, ROC-AUC i percentatge de fallades detectades amb l'1 % d'alertes. Mai l'accuracy.
  - Si s'entrena amb una mostra dels discs sense fallada, cal corregir les probabilitats amb aquesta proporció.
- **Etiquetes desconegudes:** si un disc surt de les dades abans de L+t sense fallar, `y_t` és NaN. S'exclou d'aquell horitzó, no es posa a 0.
- **Comparació de models:** les files d'un mateix disc estan correlacionades. Els intervals de confiança s'han de calcular amb bootstrap per disc, no per fila.

## Codi

- **Processament pesat** (`dataset/backblaze/scripts/`): scripts de línia d'ordres que llegeixen les dades i escriuen a `data/` o `out/`. Són llargs i no interactius, així que es queden com a scripts.
- **Anàlisi, models i figures:** notebooks de **marimo** a `notebooks/`, guardats com a `.py`.
  - Són reactius: cada variable global s'ha de definir en una sola cel·la.
  - Les variables temporals d'una cel·la van amb prefix `_`.
  - Els notebooks llegeixen sortides ja processades (`out/*.parquet`, `data/drives.parquet`) i no reprocessen els CSV.
  - Comprova'ls amb `uv run marimo check notebooks/<nom>.py`.
- **Llavors aleatòries fixes** (`random_state`, `np.random.default_rng(0)`) perquè els resultats siguin reproduïbles entre els dos membres.
- Comentaris i noms de variables en anglès. Documents (`.md`) i presentació en català.

## Git

- Fes commits petits amb missatges clars.
- No facis commit de sortides grans, de `.venv/` ni de dades.
- Si canvia el processament o hi ha resultats nous, actualitza `context_processament.md` en el mateix commit.
