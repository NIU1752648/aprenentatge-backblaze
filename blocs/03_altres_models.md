# Bloc 3. Els altres models

**Qui:** Nico (random forest, k-NN, SVM), Sergi (Naive Bayes). **Quan:** 26–29-11. **Vikunja:** #153 (random forest), #48 (Naive Bayes), #47.

## Per què més models

L'assignatura demana comparar models. Fem servir els set del temari, però no tots tenen el mateix pes:

| Model | Paper | Per què |
| --- | --- | --- |
| Gradient boosting | Principal | Probablement el que ordena millor (bloc 2) |
| Regressió logística | Principal | Senzilla, coeficients interpretables, base del model de franges (bloc 2) |
| **Random forest** | Principal | Contrast directe amb el gradient boosting: un fa la mitjana de molts arbres profunds (redueix variància), l'altre suma arbres petits en sèrie (redueix biaix) |
| Arbre de decisió | Figura | Poc profund per poder-lo llegir (bloc 5) |
| **Naive Bayes** | Taula | Bona història de teoria (vegeu més avall) |
| k-NN | Taula | Il·lustra la maledicció de la dimensionalitat |
| SVM | Taula | Nucli RBF; només amb una submostra petita |

A la presentació només hi van els tres principals; la resta, en una taula.

## Random forest (Nico)

Afegir-lo a `cv_models.py` amb dos botons: quantes variables mira a cada partició i el mínim de files per fulla. El nombre d'arbres no s'ajusta (més arbres no sobreajusten, només costen temps). Amb la mostra del bloc 1; amb totes les dades seria massa lent i ocuparia massa memòria.

## Naive Bayes (Sergi)

**La idea:** Naive Bayes calcula la probabilitat de fallar suposant que les variables són independents entre elles, i suposant una forma per a cada variable.

- El **gaussià** (el per defecte) suposa que cada variable té forma de campana. Però un comptador d'errors val 0 el 92 % de les vegades i de tant en tant 500: no té res de campana. Per això funciona malament.
- **Bernoulli:** convertim cada comptador en «és > 0, sí o no». Encaixa amb la realitat.
- **Categòric amb Laplace:** dividim cada comptador en trams (0, 1–9, 10–99, ≥ 100, absent). El suavitzat de Laplace evita que un tram que no surt mai a l'entrenament doni probabilitat zero.

**La història per a la presentació:** «de pitjor model a competitiu només canviant la forma que suposem per a les dades». I un avís: com que els comptadors pugen junts (187, 197 i 198), Naive Bayes compta la mateixa evidència diverses vegades i les probabilitats surten massa extremes, encara que ordeni bé.

**Com fer-ho:**

1. Carregar una mostra del desenvolupament de `out/landmarks/` (per exemple, totes les fallades a 30 dies i un 5 % dels discs sense fallada).
2. Preparar les variables: indicadors > 0 per a Bernoulli; trams per al categòric.
3. Entrenar `GaussianNB`, `BernoulliNB` i `CategoricalNB` (scikit-learn) amb validació creuada per disc (`GroupKFold` per `serial_number`).
4. Comparar la detecció amb l'1 % d'alertes i la PR-AUC. Es pot fer servir `budget_metrics` de `scripts/cv_models.py` per calcular la detecció.
5. Posar-ho en un notebook de marimo petit i fer commit de la taula de resultats (CSV a `out/`).

## k-NN i SVM (Nico)

Configuració senzilla i submostra (40.000 i 15.000 negatius), perquè amb totes les dades no acaben. Només van a la taula comparativa.

## Quan està fet

Cada model té una fila a la taula comparativa amb la mateixa validació i les mateixes mètriques que els principals. Abans de diumenge 29 (D2), quan es congelen els models.
