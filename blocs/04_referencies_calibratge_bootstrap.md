# Bloc 4. Referències, calibratge i bootstrap

**Qui:** Sergi (referències), Nico (calibratge i bootstrap). **Quan:** 26–29-11. **Vikunja:** #38 (referències), #45 (calibratge), #39 (bootstrap).

Són tres eines per saber si els models valen alguna cosa i quant ens en podem refiar.

## 1. Referències sense aprenentatge (Sergi)

**Per què:** dir «el model detecta un 53 % de les fallades» no diu res si una regla de dues línies ja en detecta un 46 %. Necessitem comparar amb regles que no aprenen res.

| Referència | Regla | Puntuació per ordenar els discs |
| --- | --- | --- |
| **R1** | Algun dels cinc comptadors d'error > 0 (la regla de Backblaze) | Nombre de comptadors > 0 |
| **R2** | Algun comptador ha pujat la darrera setmana | Nombre de comptadors amb `d7_*` > 0 |
| **R3** | Com de grans són els comptadors | Suma de `log1p` dels cinc comptadors |
| **Taxa per model i edat** | Què se sap sense mirar la salut | Taxa de fallada del model de disc i el tram d'edat, calculada al desenvolupament |

Els cinc comptadors són `smart_5_raw`, `smart_187_raw`, `smart_188_raw`, `smart_197_raw` i `smart_198_raw`. Els absents (NaN) compten com a 0.

**Com fer-ho:**

1. Llegir el desenvolupament de `out/landmarks/` dimecres a dimecres (només `split == "dev"`), amb les columnes necessàries.
2. Per a cada regla, calcular la puntuació de cada disc. Si hi ha empats (per exemple, molts discs amb un comptador > 0), es desempata a l'atzar amb una llavor fixa.
3. Cada dimecres, marcar el 0,1 % i l'1 % de discs amb més puntuació i comptar quantes fallades a 7, 30 i 60 dies hi ha entre els marcats. Fer-ho per a les franges també.
4. Per a la taxa per model i edat: agrupar per `model` i tram de `age_years` (per exemple, anys sencers), calcular la taxa de fallada a 30 dies, i fer servir aquesta taxa com a puntuació. **Important:** la taxa s'ha de calcular amb uns discs i avaluar amb uns altres (validació creuada per disc), o la regla es veu millor del que és.
5. Taula final: detecció amb el 0,1 % i l'1 %, PR-AUC i ROC-AUC per a cada referència. Commit del CSV a `out/` i del notebook.

**Ajuda:** `scripts/cv_models.py` té `budget_metrics` (detecció per dimecres) i `strict_labels` (etiquetes amb el tall de festes) que es poden importar.

## 2. Calibratge (Nico)

**Què és:** un model està ben calibrat si, quan diu «10 %», fallen realment un 10 % d'aquests discs. Pot ordenar molt bé i tenir probabilitats desplaçades (per exemple, dir 20 % quan és 10 %).

**Per què ens importa:** la pregunta demana una **probabilitat**, i la regla de l'aplicació (avisar si la probabilitat > 1/ρ, bloc 7) només funciona amb probabilitats calibrades. A la prova d'E5, un model que sobreestimava 1,6 vegades avisava el doble de discs del compte.

**Com:** amb les prediccions de validació creuada del bloc 2, ajustem una corba de correcció (Platt: una regressió logística sobre la puntuació). Ho comprovem amb un **diagrama de fiabilitat**: probabilitat predita contra freqüència observada; un model perfecte queda sobre la diagonal.

## 3. Bootstrap per disc (Nico)

**Per què:** si el gradient boosting detecta un 53 % i la logística un 51 %, és una diferència real o sort? El bootstrap ens dona un interval de confiança.

**Com:** repetim 1.000 vegades: sortegem discs amb reposició (el mateix disc pot sortir dues vegades), recalculem la mètrica dels dos models amb aquests discs i guardem la diferència. L'interval del 95 % són els percentils 2,5 i 97,5 de les 1.000 diferències. Sortegem **discs, no files**, perquè les files d'un mateix disc no són independents.

## Quan està fet

- Taula de referències (Sergi), comparable amb la dels models.
- Diagrama de fiabilitat abans i després de calibrar (Nico).
- Funció de bootstrap provada amb les prediccions de validació creuada, preparada per al test (Nico).
