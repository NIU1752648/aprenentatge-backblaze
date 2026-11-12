# Projecte: preguntes sobre `hair_health_dataset.csv`

Dataset: [HairSkin360 (Kaggle)](https://www.kaggle.com/datasets/krishnayadav456wrsty/hairskin360/data?select=hair_health_dataset.csv). Té 2.000 files i 38 columnes, 28 de les quals categòriques.

Els resultats són d'una primera exploració (2026-11-04): validació creuada de 5 folds, sense ajustar hiperparàmetres.

## Preguntes principals

### 1. Es pot predir la classe de pèrdua de cabell?

Fins a quin punt es pot predir `hair_loss_class` a partir de l'estil de vida, l'historial mèdic i les característiques del cabell?

| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| Classe majoritària | 46,4 % | 0,16 |
| Regressió logística | 73,7 % | 0,56 |
| SVM lineal | 73,3 % | 0,61 |
| SVM RBF | 71,5 % | 0,51 |

La diferència entre la regressió logística i l'SVM lineal entra dins de la variació entre folds.

### 2. Quins factors pesen més?

Quins grups de variables expliquen més el `hair_health_score`, i quin pes tenen els factors modificables?

| Grup de variables | R² només amb aquest grup (Ridge) |
| --- | --- |
| Mèdiques i hormonals | 0,27 |
| Antecedents familiars | 0,24 |
| Estil de vida | 0,08 |
| Edat, gènere i ètnia | 0,06 |
| Pentinat, tractaments químics i entorn | 0,05 |
| Tractament rebut | 0,02 |
| Descripció del cabell i del cuir cabellut | 0,01 |

Amb totes les variables, R² = 0,80.

### 3. Es poden detectar els casos greus?

Es poden detectar els casos `Moderate` o `Severe` (336, el 16,8 %), i quin llindar convé fer servir?

- La classe `Severe` només té 31 casos, i la regressió logística n'encerta 2.
- Agrupar `Moderate` i `Severe` permet treballar el recall, la corba PR i el desequilibri.

## Preguntes complementàries

### 4. La caiguda declarada coincideix amb la classe?

Sembla que no: l'associació entre `hair_fall_level` i `hair_loss_class` és gairebé nul·la (V de Cramér = 0,03). Fins i tot entre qui declara una caiguda `Severe`, el 36 % és `No_Hair_Loss`.

### 5. Els antecedents familiars afecten igual tothom?

El score baixa 12,3 punts en dones i 17,5 en homes. Falta comprovar si aquesta diferència és significativa i si cal un model amb interaccions.

### 6. Els tractaments funcionen? (per a la discussió)

El `No_Hair_Loss` és del 48 % amb minoxidil i del 31 % sense tractament, però això no demostra que el minoxidil funcioni. Permet explicar la diferència entre predir i establir una causa.

## Variables que cal excloure

| Variable | Motiu | Efecte en la regressió logística |
| --- | --- | --- |
| `hair_health_score` | La classe surt directament d'aquest valor, amb talls a 75, 55 i 35 | L'accuracy puja del 74 % al 94 % |
| `age_of_onset_years` | Només està buida en els 736 casos `No_Hair_Loss` | Si el model pot saber que falta, l'accuracy puja del 74 % al 89 % |
| `treatment_history` | El tractament probablement arriba després de la pèrdua | Treure'l baixa l'accuracy del 73,7 % al 72,5 % |

Aquestes exclusions són per a la predicció de la classe. A la pregunta 6, `treatment_history` és justament la variable que s'estudia.

## Limitacions

- **Dades probablement sintètiques (sense confirmar):** no hi ha cap duplicat, l'inici sempre és entre 1 i 11 anys abans de l'edat actual, i la caiguda declarada no té relació amb la classe. Si és així, les conclusions descriuen com s'han generat les dades, no la biologia del cabell, i caldrà dir-ho a l'informe.
- **Objectius poc útils:** `hair_fall_level`, `hair_density`, `stress_level`, `family_history`, `medical_condition` i `treatment_history` queden pràcticament a l'atzar si s'intenten predir.
- **Models pendents de comparar:** k-NN (53 %) i Naive Bayes gaussià (17 %) encara no són comparables. Amb variables one-hot, Naive Bayes gaussià és una mala elecció i caldria provar `CategoricalNB`.

## Pendent

- Preguntar al professor si accepta un dataset que sembla sintètic.
