# Bloc 8. Notebooks i figures finals

**Qui:** Nico, amb les figures de Sergi (arbre, Naive Bayes, referències, Kaplan-Meier). **Quan:** 2–7-12. **Vikunja:** #49, #57, #58.

## Per què notebooks

Els scripts (`scripts/`) fan la feina pesada i escriuen resultats a `out/`. Els **notebooks de marimo** (`notebooks/`) només llegeixen aquests resultats i fan les taules i figures de la presentació. Així qualsevol dels dos pot regenerar una figura en segons, sense tornar a entrenar res.

## Quins notebooks

| Notebook | Contingut | De quins blocs |
| --- | --- | --- |
| `01_dades.py` (fet) | Dades, calendari, festes | — |
| `01b_etiquetes.py` (fet) | Error d'etiqueta | — |
| `02_validacio.py` | Partició, per què validem per disc, validació contra test | 2, 6 |
| `03_models.py` | Corbes de validació per hiperparàmetre, taula dels set models, franges contra horitzons (E4), corba d'aprenentatge | 1, 2, 3, 4 |
| `04_senyal.py` | Escala de variables (E1), discs i fallades nets (E2), arbre, permutació, SHAP | 5 |
| `05_aplicacio.py` | Pressupost, estalvi, urgència, antelació, calibratge, Kaplan-Meier | 4, 7 |

## Regles de marimo (per si en toques un)

- Cada variable global es defineix en una sola cel·la (marimo és reactiu: si canvies una cel·la, es recalculen les que en depenen).
- Les variables temporals d'una cel·la porten el prefix `_`.
- Es guarden com a `.py` i es comproven amb `uv run marimo check notebooks/<nom>.py`.
- S'obren amb `uv run marimo edit notebooks/<nom>.py`.

## Les figures

- Es desen a `out/figures/` en PNG (es versionen).
- Text en català, eixos amb unitats, llegenda quan hi ha més d'una sèrie.
- Mai dos eixos verticals a la mateixa figura: si hi ha dues mesures, dues figures.

## Quan està fet

Cada diapositiva té la seva figura generada per un notebook que funciona amb `marimo check` sense errors.
