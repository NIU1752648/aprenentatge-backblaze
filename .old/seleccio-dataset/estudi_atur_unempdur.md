# Estudi de viabilitat: UnempDur / `jobless` (durada de l'atur, McCall 1996)

**Veredicte: SÍ que encaixa, amb reserves. L'estructura (durada + censura) és ideal per a l'enfocament d'horitzons i finestres temporals, i McCall no va fer predicció. Però el senyal és modest (ROC-AUC ≈ 0,65–0,68 sense leakage), les dades són dels anys vuitanta i l'horitzó de 52 setmanes no és utilitzable. Cal fer servir la versió de 43 variables (`camerondata::jobless`), no la d'11 de `Ecdat`.**

## 1. Resum del dataset
- **Font:** CPS Displaced Workers Supplements (gener 1986, 1988, 1990 i 1992). Treballadors de 20 a 61 anys que van perdre la feina per tancament, falta de feina o supressió del lloc. Prové de McCall (1996), *Econometrica* 64, 647–682.
- **Còpies descarregades:** `Ecdat::UnempDur` (CSV de Rdatasets: 3.343 × 11, verificat) i `camerondata::jobless` (fitxer `.rda` del tarball de CRAN v1.0.0: 3.343 × 43). Les files coincideixen una a una (`spell` és idèntic al 100 %).
- **Variables d'UnempDur:** `spell` (durada en períodes de 2 setmanes, d'1 a 28), `censor1–4`, `age`, `ui`, `reprate`, `disrate`, `logwage`, `tenure`.
- **Variables que afegeix `jobless`:** motiu de l'acomiadament (`slack`, `abolpos`), `explose` (s'esperava l'acomiadament), `stateur` (atur estatal), dades demogràfiques (`female`, `married`, `child`, `nonwhite`, escolaritat…), sector, coll blau, `smsa`, any de la pèrdua i regió. **Les variables extra són importants:** amb les 5 variables d'UnempDur sense `ui`, la regressió logística només arriba a un AUC de 0,55–0,60.

## 2. Construcció del target
- **Esdeveniment:** reocupació (`censor1`, `censor2` o `censor3`). Sense esdeveniment, l'observació és censurada (`censor4`). Hi ha **102 files amb tots els `censor` a 0**: no estan documentades i les he tractat com a censurades.
- Per a un horitzó *h* (en períodes de 2 setmanes): y=1 si `spell` ≤ h i hi ha esdeveniment; y=0 si `spell` > h; desconegut si la persona va ser censurada abans de *h*. Les files desconegudes s'exclouen de l'entrenament però sí que es fan prediccions per a elles.

| h (períodes de 2 setmanes) | reocupats (y=1) | encara aturats (y=0) | censurats (desconegut) | utilitzables | taxa d'esdeveniment |
|---|---|---|---|---|---|
| 4 (8 s) | 1.213 | 1.676 | 454 | 2.889 | 0,42 |
| 6 (12 s ≈ 3 mesos) | 1.469 | 1.196 | 678 | 2.665 | 0,55 |
| 13 (26 s) | 1.826 | 415 | 1.102 | 2.241 | 0,82 |
| 26 (52 s) | 1.977 | **33** | 1.333 | 2.010 | 0,98 → **inviable** |

- **Desenllaços competidors:** 1.073 persones van trobar feina a temps complet, 339 a temps parcial i 574 de manera desconeguda (per a 3 de cada 10 reocupats no se sap el tipus). Es podria fer un model multiclasse per horitzó (complet / parcial / encara aturat), però la categoria "desconeguda" en complica la interpretació. Ho proposo només com a discussió.

## 3. Resultats (CV de 5 particions, prediccions out-of-fold; variables `jobless` **sense `ui`**)
Hiperparàmetres fixos i **sense ajustar** (C=1, k=50, `ccp_alpha`=0,002, RF amb 500 arbres i `min_samples_leaf`=5). L'AUC i el PR-AUC es calculen sobre les files utilitzables.

| Model | AUC h=4 | AUC h=6 | AUC h=13 | Brier h=4 | Brier h=6 | Brier h=13 |
|---|---|---|---|---|---|---|
| Baseline (prevalença) | 0,500 | 0,500 | 0,500 | 0,244 | 0,247 | 0,151 |
| Logística (L2) | **0,651** | **0,665** | 0,676 | **0,228** | 0,228 | 0,142 |
| Logística L1 (C=0,1) | 0,648 | 0,663 | **0,679** | 0,228 | **0,227** | **0,141** |
| k-NN (k=50) | 0,613 | 0,627 | 0,634 | 0,235 | 0,235 | 0,146 |
| Naive Bayes | 0,601 | 0,623 | 0,618 | 0,292 | 0,293 | 0,212 |
| SVM lineal (calibrat) | 0,625 | 0,628 | 0,614 | 0,239 | 0,243 | 0,148 |
| SVM RBF | 0,619 | 0,646 | 0,645 | 0,234 | 0,231 | 0,145 |
| Arbre CART (podat) | 0,554 | 0,590 | 0,580 | 0,250 | 0,247 | 0,152 |
| Random forest | 0,626 | 0,648 | 0,658 | 0,232 | 0,231 | 0,143 |
| *Temps discret, logística* | 0,656 | 0,665 | 0,688 | 0,228 | 0,232 | 0,161 |
| *Logística **amb** `ui` (leakage)* | *0,777* | *0,765* | *0,739* | *0,186* | *0,195* | *0,136* |

- PR-AUC de la logística: 0,56 / 0,69 / 0,89, davant d'una baseline de 0,42 / 0,55 / 0,82.
- Les millores sobre la baseline són reals però petites. La logística iguala o supera els models no lineals. L'error OOB del RF (accuracy 0,61 a h=6) és coherent amb la CV.
- **Importància a h=6:** per impuresa, guanyen les variables contínues (`age`, `stateur`, `logwage`, `disrate`). Per permutació, el rànquing és `stateur`, `age`, `slack`, `nonwhite` i `tenure`. És un bon exemple del biaix de la importància per impuresa.

## 4. Probabilitat acumulada en el temps
He entrenat models **separats** per a h=1…20 i un model de **temps discret**: un dataset persona-període de 20.887 files, amb l'interval com a variable (dummies per a la logística, numèric per al RF) i F(t) = 1 − ∏(1 − ĥ_k). He comparat la mitjana de les F prediccions amb Kaplan-Meier (`lifelines`).

| setmanes | 8 | 12 | 26 | 40 |
|---|---|---|---|---|
| Kaplan-Meier | 0,375 | 0,473 | 0,662 | 0,799 |
| Temps discret (logística) | 0,377 | 0,475 | 0,663 | 0,800 |
| Temps discret (RF) | 0,394 | 0,502 | 0,727 | 0,850 |
| Models separats (logística) | 0,417 | 0,546 | 0,800 | 0,936 |

- Els **models separats sobreestimen** la probabilitat de reocupació (+14 punts a 26 setmanes). Com que s'entrenen excloent els censurats abans de *h*, que són persones encara aturades, la mostra queda esbiaixada cap a l'esdeveniment. El model de temps discret aprofita la informació parcial dels censurats i **reprodueix Kaplan-Meier gairebé exactament** en el cas de la logística.
- **No-monotonia:** amb models separats, el 83 % (logística) i el 93 % (RF) de les persones tenen alguna corba F(t) que baixa en algun punt entre h=1 i h=20. Amb temps discret, el 0 %, per construcció.
- En capacitat de discriminació, els dos enfocaments estan empatats (AUC similar). El Brier del temps discret a h=13 és pitjor (0,161 vs 0,142), però en part és un artefacte d'avaluar només sobre les files no censurades, que és la mateixa mostra esbiaixada.

## 5. Aplicació pràctica
En el moment de la pèrdua de la feina, el model estimaria la probabilitat que la persona **continuï aturada als 3 i als 6 mesos** (1 − F). Els serveis d'ocupació podrien prioritzar orientació i formació per a les persones amb més risc d'atur de llarga durada ("profiling" estadístic, com el WPRS dels EUA). Cal discutir el llindar i la capacitat del servei, per exemple quants casos es poden atendre amb una precisió donada (corba PR). També cal parlar d'equitat: variables com `nonwhite` o `female` poden aportar predicció, però el seu ús és qüestionable.

## 6. Encaix amb el temari
Logística amb Ridge/Lasso, k-NN, NB, SVM lineal i RBF, CART amb `ccp_alpha`, RF (OOB, importància per impuresa vs permutació) i validació creuada (al persona-període cal agrupar per persona). També cobreix ROC/PR, Brier i calibratge, i un cas clar de leakage (`ui`). El biaix de selecció per censura es pot explicar visualment. Kaplan-Meier només s'usa com a referència.

## 7. Problemes i riscos
- **Leakage amb `ui`:** "ha sol·licitat la prestació" es decideix *després* de quedar-se a l'atur. Ho confirma que el 31 % de les persones sense `ui` troben feina al primer període, davant del 4,5 % de les que en tenen. Incloure-la fa pujar l'AUC de 0,65 a 0,78, de manera enganyosa. **Cal excloure-la.** `reprate` i `disrate` són taxes "elegibles", calculades amb les regles, i en principi són prèvies a l'atur (no verificat en detall).
- **Dades antigues** (pèrdues de feina entre 1985 i 1991, als EUA): la utilitat real actual és limitada, però el projecte es pot plantejar com a prova de concepte.
- **Senyal feble:** l'AUC màxim és ≈ 0,68 i no hi ha variables de recerca de feina ni d'historial laboral.
- **Censura elevada:** el 41 % de les observacions són censurades, i l'horitzó de 52 setmanes és inviable. La durada màxima és de 28 períodes i no sé si hi ha topcoding.
- **"Heaping" a `spell`:** els valors senars (1, 3, 5, 7…) són molt més freqüents que els parells, segurament per l'arrodoniment de les respostes. Això fa que les corbes tinguin esglaons.
- Hi ha 102 files sense cap indicador de censura, i la categoria `censor3` és ambigua.

## 8. Comprovació de l'article dels creadors
McCall (1996) estima un **model estructural de recerca de feina i un model de riscos competidors** (temps complet vs temps parcial), amb riscos correlacionats i coeficients que varien en el temps. L'objectiu és la **inferència causal** de l'efecte del *disregard* de la prestació sobre la reocupació a temps parcial, i no avaluar la predicció individual. Segons l'abstract, que és l'única part consultada perquè el text complet està restringit a JSTOR, l'article no inclou classificació per horitzons ni validació predictiva. **No compta com "la mateixa anàlisi".** Cameron i Trivedi (2005) fan servir les dades per il·lustrar models de durada en el seu llibre de text: són compiladors, no creadors, i també fan inferència. En una cerca ràpida no he trobat cap treball de McCall sobre predicció o profiling amb aquestes dades.

## 9. Què s'ha comprovat i què no (2026-11-09)
- **Comprovat:** descàrrega i inspecció del CSV de Rdatasets (Ecdat) i del `jobless.rda` de CRAN (camerondata 1.0.0), i la documentació de les dues versions. També el recompte per horitzó, els 8 models amb CV de 5 particions (amb i sense `ui`, i amb 5 vs 42 variables), el model de temps discret (logística i RF), Kaplan-Meier, la no-monotonia i les importàncies del RF.
- **No comprovat:** el text complet de McCall (1996), només l'abstract i els metadades. L'ajust d'hiperparàmetres, que no s'ha fet. Tampoc s'han comprovat el significat exacte de les 102 files sense indicador, la temporalitat de `reprate` i `disrate`, l'existència de topcoding a 28 períodes ni el model multiclasse de desenllaços competidors. La CV no està estratificada per any de la pèrdua.
- Els scripts i resultats són al scratchpad de la sessió (`unempdur/scripts/main.py`, `unempdur/out/results.csv`, `curves.csv`), que no és un directori persistent.
