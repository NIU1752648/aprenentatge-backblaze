# Predicció de fallades de discs durs (Backblaze Drive Stats)

Treball en parella de l'assignatura d'Aprenentatge Computacional. L'enunciat és a [`enunciat_treball.md`](enunciat_treball.md) (original a `enunciat_treball.pdf`).

Fem servir les dades diàries de discs durs de Backblaze (valors SMART i fallades) per predir si un disc fallarà en els propers 7, 30 o 60 dies. Comparem diversos models i estudiem quant del senyal prové dels comptadors SMART que la mateixa Backblaze fa servir per decidir que un disc ha fallat.

## Documents

| Fitxer | Contingut |
| --- | --- |
| [`dataset/estudi_backblaze.md`](dataset/estudi_backblaze.md) | Estudi de viabilitat: dataset, objectiu, primers resultats i riscos |
| [`dataset/backblaze/context_processament.md`](dataset/backblaze/context_processament.md) | Com estan processades les dades, què fa cada script i què queda pendent |
| [`AGENTS.md`](AGENTS.md) | Convencions del repositori, per a nosaltres i per als assistents de codi |
| `.old/seleccio-dataset/` | Datasets que vam considerar abans de triar Backblaze |

Si feu servir Claude Code, creeu un `CLAUDE.md` local (no es versiona) amb la línia `@AGENTS.md` perquè llegeixi les convencions del projecte.

## Posada en marxa

### 1. Entorn de Python

Les llibreries es gestionen amb [uv](https://docs.astral.sh/uv/). Cadascú té el seu entorn `.venv/` local, que no es versiona.

```bash
# Si no tens uv: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync            # crea .venv/ amb les versions exactes de uv.lock
```

- **Afegir una llibreria:** `uv add <paquet>`. Cal fer commit de `pyproject.toml` i de `uv.lock`.
- **Executar un script:** `uv run python ...`. No cal activar l'entorn.

### 2. Dades

Les dades no són al repositori (14 GB per trimestre en CSV). Per obtenir-les:

1. Descarrega el zip del trimestre des de la pàgina [Backblaze: Hard Drive Data and Stats](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data), per exemple `data_Q4_2025.zip`.
2. Descomprimeix cada trimestre en un directori nou:

   ```bash
   cd dataset/backblaze
   mkdir -p data/q4 && unzip data_Q4_2025.zip -d data/q4
   ```

3. Consolida'l a Parquet (uns 3 minuts per trimestre, ~900 MB de sortida):

   ```bash
   uv run python -I scripts/consolidate.py data/q4 data 2025Q4
   ```

   Això crea `data/panel/2025Q4.parquet` (una fila per disc i dia) i reconstrueix `data/drives.parquet` (una fila per disc) amb tots els trimestres de `data/panel/`. Un cop consolidat, els CSV i el zip es poden esborrar.

### 3. Taula de dates de referència i models

Des de `dataset/backblaze/`:

```bash
uv run python -I scripts/build.py data out/landmarks.parquet   # taula de dates de referència i etiquetes y_7, y_30, y_60
uv run python -I scripts/explore.py out/landmarks.parquet      # recomptes i regla de Backblaze
uv run python -I scripts/model.py out/landmarks.parquet out     # models i resultats a out/
```

Els resultats petits (CSV i logs de `out/`) estan versionats. Els `.parquet` de `out/` no hi són perquè es poden regenerar.

### 4. Notebooks de marimo

L'anàlisi i les figures per a la presentació es faran amb notebooks de [marimo](https://marimo.io/). Es guarden com a fitxers `.py` normals a `notebooks/`.

```bash
uv run marimo edit notebooks/<nom>.py    # editar
uv run marimo run notebooks/<nom>.py     # veure com a aplicació, sense codi
```

## Estructura

```
projecte/
├── README.md, AGENTS.md
├── pyproject.toml, uv.lock, .python-version   # entorn (uv)
├── enunciat_treball.{pdf,md}
├── notebooks/                    # notebooks de marimo (anàlisi i presentació)
├── dataset/
│   ├── estudi_backblaze.md
│   └── backblaze/
│       ├── context_processament.md
│       ├── scripts/              # processament: consolidate, build, explore, model
│       ├── data/                 # (no versionat) CSV, panel/*.parquet, drives.parquet
│       └── out/                  # resultats; els *.parquet no es versionen
└── .old/                         # selecció de dataset (històric)
```

## Terminis

- **13 de desembre, 23:59:** entrega al Campus Virtual de la presentació en PDF i d'un zip amb el codi.
- **14 i 16 de desembre:** presentació de 8 minuts, repartits a parts iguals entre els dos membres.

## Llicència de les dades

Les dades són de Backblaze. Es poden fer servir lliurement si se cita la font i no es venen.
