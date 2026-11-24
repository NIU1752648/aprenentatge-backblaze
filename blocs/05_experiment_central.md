# Bloc 5. Experiment central: de què ens podem refiar

**Qui:** Nico (E1, E2, permutació, SHAP), Sergi (arbre dibuixat). **Quan:** 30-11 – 3-12. **Vikunja:** #54 (E1), #55 (E2), #56 (permutació i SHAP), #155 (arbre), #58 (notebook 04).

## La pregunta d'aquest bloc

Backblaze decideix que un disc ha fallat **en part mirant els cinc comptadors d'error SMART**. Si el nostre model fa servir aquests mateixos comptadors, pot semblar molt bo només perquè ha après la regla de Backblaze, no perquè sàpiga anticipar fallades. Aquest bloc mesura **quina part del que fa el model és la regla i quina part és anticipació de veritat**. És el que fa el treball original.

## E1. Escala de variables

Entrenem el mateix model (logística i el millor conjunt d'arbres) afegint grups de variables un a un, i mirem quant guanya cada pas a cada franja:

| Pas | Variables | Què vol dir |
| --- | --- | --- |
| G0 | Fabricant, model, capacitat, edat | El que se sap sense mirar la salut |
| + G1 | Altres SMART (temperatura, encesos...) | Salut general que no és la regla |
| + GO | Dies absent, dies des que torna... | Senyal operatiu (pot reflectir sospites de Backblaze) |
| + G2 | Valor dels cinc comptadors d'error | La regla de Backblaze |
| + G3 | Canvis dels comptadors en 1, 7 i 30 dies | Avís recent |

Si quasi tot el guany arriba amb G2 i G3, el model és sobretot la regla. Si G0, G1 i GO ja aporten força, hi ha senyal que la regla no veu.

## E2. Discs nets i fallades netes

- **Discs nets:** els que tenen els cinc comptadors a zero (un 92 % del parc). Aquí la regla de Backblaze no pot dir res. Si el model hi detecta fallades millor que la taxa per model i edat (bloc 4), està anticipant de veritat.
- **Fallades netes:** fallades d'un disc sense cap error l'últim dia (un 24–27 %). No poden venir de la regla. Quantes en detecta el model?

## Importància per permutació i SHAP

Dues maneres de veure en què es basa el model:

- **Permutació:** barregem a l'atzar els valors d'una variable i mirem quant empitjora el model. Si empitjora molt, la variable era important.
- **SHAP** (tema Explainability del curs): reparteix cada predicció entre les variables, com un premi entre jugadors (valors de Shapley). Com que les aportacions sumen la predicció, les sumem per grups (G0...G3): per a cada alerta, quina part del risc ve dels comptadors d'error? Gràfics: un per franja (els comptadors pesen menys a 60 dies?) i l'explicació d'unes quantes alertes reals.

**Límit a dir:** les variables que porten la mateixa informació (el valor d'un comptador i quant ha pujat) es reparteixen el mèrit; per això mirem grups i no variables soltes. I SHAP explica el model, no per què fallen els discs.

## Arbre poc profund dibuixat (Sergi)

**Per què:** un arbre de 2 o 3 nivells es pot llegir sencer en una diapositiva. Si la primera pregunta que fa és «smart_5 > 0?» o «d7_197 > 0?», ha redescobert la regla de Backblaze sol. És una figura molt clara per explicar el problema.

**Com fer-ho:**

1. Agafar una mostra del desenvolupament de `out/landmarks/` (totes les fallades a 30 dies i un 5 % dels discs sense fallada, amb `y_30` com a etiqueta).
2. Entrenar `DecisionTreeClassifier(max_depth=3, min_samples_leaf=…, random_state=0)` amb les variables de tots els grups. Els NaN s'han d'omplir (per exemple, amb 0 per als comptadors i la mediana per a la resta).
3. Dibuixar-lo amb `sklearn.tree.plot_tree` i desar el PNG a `out/figures/`.
4. Escriure dues o tres frases: quines preguntes fa, i si coincideixen amb la regla.
5. Opcional: el mateix sense els cinc comptadors (G0 + G1 + GO). Quines preguntes fa llavors?

## Quan està fet

Taula d'E1 per franja, resultat d'E2 en dues frases, figures de SHAP i l'arbre, tot al notebook `04_senyal.py`. És el resultat central de la presentació.
