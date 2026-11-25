# Posada al dia del projecte (23-11-2026)

Aquest document resumeix, amb paraules senzilles, què estem fent, per què ho fem així, què tenim fet i què ve ara. És per posar-nos d'acord tots dos. Els detalls tècnics són a `pla_projecte.md` (el pla sencer) i a `dataset/backblaze/context_processament.md` (l'estat del processament).

---

## 1. De què va el projecte

### La pregunta

L'assignatura ens demana triar un dataset real, fer-nos una pregunta i respondre-la amb experiments de machine learning. La pregunta que vam enviar al professor és:

> *A partir de l'estat SMART d'un disc en un moment donat, quina probabilitat hi ha que falli en els propers t dies, quina esperança podem donar a aquests discs i com podem fer servir aquesta informació per saber quins discs s'han de substituir i quan?*

Té tres parts:

1. **Probabilitat:** quina probabilitat té un disc de fallar en els propers 7, 30 i 60 dies.
2. **Esperança:** quant temps els queda, més o menys, als discs que veiem en risc.
3. **Aplicació:** quins discs s'han de canviar i amb quina urgència.

### Les dades

**Backblaze** és una empresa que guarda dades al núvol i té centenars de milers de discs durs funcionant. Cada dia publica una fila per a cada disc, amb:

- El model i la capacitat del disc.
- Els valors **SMART**: comptadors que el mateix disc porta de si mateix. Per exemple, hores encès, temperatura, o sectors que han donat error i s'han hagut de reubicar.
- Una columna `failure`, que val 1 el dia que Backblaze dona el disc per mort i el retira.

Fem servir **sis trimestres**, de gener de 2025 a juny de 2026: 546 dies, uns 372.000 discs diferents i unes 6.900 fallades. Són uns 180 milions de files.

### Per què és un problema difícil

- **Les fallades són molt rares.** Cada dia falla, de mitjana, un disc de cada 25.000. En 30 dies en falla un de cada 900, aproximadament. Un model que digués «cap disc fallarà» encertaria el 99,9 % de les vegades i no serviria per a res. Per això **mai fem servir l'accuracy** (el percentatge d'encerts) i mirem altres mètriques (secció 4).
- **Una mateixa màquina surt molts dies.** Un disc apareix cada dia durant mesos, i les files d'un mateix disc s'assemblen molt. Si el mateix disc surt a l'entrenament i a la prova, el model «el reconeix» i sembla millor del que és. Per això sempre separem **per disc**.
- **El temps importa.** Volem predir el futur, així que hem de provar el model amb dades posteriors a les que ha vist, com passaria a la vida real.

---

## 2. Com hem plantejat el problema (i per què)

### Dates de referència

No podem entrenar directament amb la taula dia a dia. Fem com faria un tècnic de manteniment: **cada dimecres** mira tots els discs i decideix quins vigilar. Cada dimecres és una **data de referència** (L).

Per a cada dimecres i cada disc que funciona aquell dia guardem dues coses:

- **El que se sap aquell dia** (les variables): els valors SMART, quant han canviat les últimes setmanes, l'edat del disc, si ha faltat alguns dies...
- **El que passa després** (l'etiqueta): si el disc falla en els 7, 30 o 60 dies següents.

Així convertim el problema en una **classificació binària** del curs: donades les variables, el disc fallarà abans de t dies, sí o no?

**Per què dimecres:** les fallades es registren menys els diumenges i els dilluns, perquè depenen de quan treballa la gent del centre de dades. Si sempre fem servir el mateix dia de la setmana, aquest efecte no barreja les dates.

### Diversos horitzons i franges de temps

El professor ens va demanar fixar un **horitzó t** i entrenar **diversos models per horitzó o per franja de temps**, per obtenir una probabilitat acumulada. Ho fem així:

- **Tres franges:** de 0 a 7 dies, de 7 a 30 i de 30 a 60.
- **Un model per franja.** Cadascun respon «si el disc arriba viu a l'inici de la franja, quina probabilitat té de fallar-hi?».
- **Probabilitat acumulada:** es combinen les tres. Per exemple, P(fallar en 30 dies) = 1 − P(sobreviure a la franja 1) × P(sobreviure a la franja 2).

**Per què franges i no un model per a cada horitzó:**

- Amb models separats per a 7, 30 i 60 dies, la probabilitat a 60 dies pot sortir **més petita** que la de 30, cosa que no té sentit. A l'estudi previ passava fins a un 27 % de les vegades.
- Amb franges, la corba sempre puja.
- També ens diu **quan** és més probable que falli el disc, que és exactament la part «i quan?» de la pregunta.
- Els models separats per horitzó es mantenen com a comparació.

### Separar entrenament i prova en el temps

Dividim les dates en tres blocs:

```
2025                                                    2026
G  F  M  A  M  J  J  A  S  O  N  D  |  G  F  M  A  M  J
   [------ desenvolupament ------]      [---- test ----]
   5-2                         5-11     7-1         29-4
                                 <- separació ->
```

| Bloc | Dates | Per a què serveix |
| --- | --- | --- |
| **Desenvolupament** | 40 dimecres, del 5-2 al 5-11-2025 | Entrenar models, provar variables, triar hiperparàmetres |
| **Separació** | 8 dimecres, del 12-11 al 31-12-2025 | No es fan servir |
| **Test** | 17 dimecres, del 7-1 al 29-4-2026 | Avaluació final, **una sola vegada**, al final |

**Per què la separació:** l'etiqueta de l'última data de desenvolupament mira 60 dies endavant. Si el test comencés just després, el model hauria «vist» fallades del període de test a través de les etiquetes. Els 63 dies de separació ho eviten.

**Per què el test només s'executa una vegada:** si anem mirant el test i retocant el model, l'acabem ajustant al test sense voler, i el resultat ja no diu com funcionaria amb dades noves. Tot es decideix amb el desenvolupament, i el test es fa servir només al final (previst el 4-12).

**Dins del desenvolupament** fem validació creuada amb `GroupKFold` per disc (3 grups). Cada disc queda en un sol grup, de manera que mai no apareix alhora a l'entrenament i a la validació.

### L'angle propi: de què ens podem refiar

Hi ha un problema important amb aquestes dades. **Backblaze decideix que un disc ha fallat, en part, mirant cinc comptadors SMART** (5, 187, 188, 197 i 198, que compten sectors defectuosos i errors). Si el model fa servir aquests mateixos comptadors, pot semblar molt bo només perquè ha après la regla que fa servir Backblaze, no perquè sàpiga anticipar res.

Per això el treball no és «un classificador més». Mesurem **quina part del que prediu el model és la regla de Backblaze i quina part és anticipació de veritat**:

- **Escala de variables:** entrenem el mateix model afegint grups de variables un a un (secció 3) i mirem quant guanya cada grup a cada franja.
- **Discs nets i marcats:** avaluem per separat els discs amb els cinc comptadors a zero (on la regla no pot dir res) i els que ja en tenen algun per sobre de zero.
- **Fallades netes i brutes:** separem les fallades segons si el disc tenia errors l'últim dia o no. Les que no en tenien no poden venir de la regla.

Aquesta és la part que fa el treball original. A més, el professor no vol que repetim anàlisis que ja hagi publicat Backblaze.

---

## 3. Què tenim fet

### Processament de les dades (fet)

Cada pas és un script de `dataset/backblaze/scripts/` que llegeix un fitxer i n'escriu un altre:

| Pas | Script | Què fa | Resultat |
| --- | --- | --- | --- |
| 1 | `consolidate.py` | Converteix els CSV de cada trimestre (12 GB) a Parquet (0,9 GB) i treu les columnes SMART gairebé buides | `data/panel/` (una fila per disc i dia) i `data/drives.parquet` (una fila per disc) |
| 2 | `daily_stats.py` | Fa recomptes diaris (discs actius, fallades, errors...) per estudiar el calendari | `out/daily_stats.parquet` i altres |
| 3 | `presence.py` | Apunta quins dies apareix cada disc i detecta els dies que falten molts discs alhora | `data/presence.parquet` i `data/presence_days.parquet` |
| 4 | `build.py` | Construeix la taula de dates de referència: variables i etiquetes per a cada dimecres | `out/landmarks/L=<data>.parquet`, un fitxer per dimecres |

**Format de les dades:**

- **Per què Parquet:** ocupa molt menys que el CSV i permet llegir només algunes columnes o algun dia sense carregar-ho tot. La taula sencera no cap a la memòria d'un portàtil.
- **On és tot explicat:** `dataset/backblaze/data/README.md` explica què hi ha a cada fitxer.

### La taula de dates de referència

És la taula amb què entrenarem. Té 65 dimecres i 21,3 milions de files (un disc en una data per fila):

| Bloc | Dates | Files | Discs que fallen en 7 / 30 / 60 dies (files) |
| --- | --- | --- | --- |
| Desenvolupament | 40 | 12,8 M | 3.152 / 14.426 / 28.178 |
| Separació | 8 | 2,7 M | — |
| Test | 17 | 5,8 M | 1.399 / 6.509 / 13.374 |

Les variables estan organitzades en grups, que són els passos de l'escala de variables:

| Grup | Què conté | Per què |
| --- | --- | --- |
| **G0 Context** | Fabricant, model, capacitat, edat | El que se sap sense mirar la salut del disc |
| **G1 Altres SMART** | Temperatura, encesos, hores de càrrega... | Salut general que no és la regla de Backblaze |
| **GO Senyal operatiu** | Dies que el disc ha faltat el darrer mes, dies des que va tornar, dies des que el veiem | Un disc que desapareix i torna sovint està tenint problemes. Però pot ser que Backblaze ja sospités d'ell, per això va a part |
| **G2 Comptadors d'error** | Valor actual dels 5 comptadors | És la regla de Backblaze |
| **G3 Canvis dels comptadors** | Quant han pujat en 1, 7 i 30 dies | Un comptador que puja de cop és un avís recent |

**Etiquetes:** per a cada fila, si el disc falla en 7, 30 o 60 dies (1 o 0). Si el disc desapareix de les dades abans d'acabar l'horitzó sense fallar, no sabem què li passa: l'etiqueta queda **buida (NaN)** i aquella fila no s'utilitza per a aquell horitzó. No la posem a 0, perquè seria inventar-nos que no ha fallat.

### Què hem après de les dades

**1. Les festes amaguen fallades.** De Thanksgiving (17-11-2025) a Reis, es registren un 57 % menys de fallades.

- Els discs es deterioren igual: els comptadors d'error continuen pujant.
- El que passa és que a les festes hi ha menys gent i **es registren menys fallades**. Després de Reis el ritme torna a pujar (nivell normal a partir del març), però les fallades que falten no apareixen més tard.
- Per tant, les etiquetes dels dimecres que miren cap a les festes tenen fallades que «falten».
- Decisió (24-11, comprovada amb el criteri del professor el 25-11): tallar les etiquetes de desenvolupament el 16-11-2025, de manera que cap etiqueta miri més enllà d'aquell dia.

**2. Una regla senzilla ja funciona força bé.** «Algun comptador d'error ha pujat» ja troba un 46 % de les fallades a 30 dies, mirant només l'1 % de discs amb més risc. El primer model de prova (gradient boosting) en troba un 52 %.

- Les fallades sense cap error previ (un 24 %) gairebé no les troba ningú.
- Treient els comptadors d'error del model, se'n troba un 27 %: és el senyal que no ve de la regla.
- Això confirma que l'angle propi té sentit.

**3. Els discs que desapareixen uns dies són sospitosos.** Els discs que han faltat algun dia el darrer mes fallen unes 9 vegades més que la resta.

**4. Hi va haver una caiguda de dades el 12-5-2025.** Aquell dia falta un 22 % dels discs, però no són absències reals d'aquests discs. Aquests dies no els comptem com a absència.

---

## 4. Com mesurarem si un model és bo

Com que les fallades són tan rares, fem servir mètriques pensades per a l'aplicació:

- **Detecció amb un pressupost d'alertes (la principal).** Cada setmana, l'equip de manteniment només pot revisar un nombre limitat de discs, per exemple l'1 % amb més risc. La mètrica és el percentatge de les fallades reals que són dins d'aquest 1 %. Respon directament «quins discs s'han de canviar».
- **PR-AUC:** resumeix com de bé ordena el model els discs que fallaran, tenint en compte que són pocs. És millor que la ROC-AUC quan les classes estan tan desequilibrades.
- **ROC-AUC:** la mostrem com a secundària, perquè és la que coneix tothom.
- **Calibratge:** si el model diu un 10 %, han de fallar realment un 10 % d'aquests discs. Ho necessitem perquè la pregunta demana una **probabilitat**, no només un ordre.

**Referències sense aprenentatge.** Per saber si un model aporta alguna cosa, el comparem amb regles que no aprenen res:

- R1: algun comptador d'error > 0 (la regla de Backblaze).
- R2: algun comptador ha pujat la darrera setmana.
- R3: suma dels comptadors.
- La taxa de fallada per model de disc i edat.

**Intervals de confiança.** Per dir si un model és millor que un altre, fem **bootstrap per disc**: repetim l'avaluació 1.000 vegades sortejant discs (no files) i mirem si la diferència es manté. Sorteig discs perquè les files d'un mateix disc no són independents.

---

## 5. Models

Només fem servir models del temari:

- Regressió logística.
- k-NN.
- Naive Bayes.
- Arbre de decisió.
- Random forest.
- SVM.
- Gradient boosting.

A la presentació hi aniran la regressió logística, el random forest i el gradient boosting, i un arbre poc profund dibuixat, per veure si redescobreix la regla de Backblaze. La resta aniran en una taula.

**Explicabilitat amb SHAP.** SHAP és del tema Explainability del curs. Reparteix cada predicció entre les variables, com si fossin jugadors que es reparteixen un premi (valors de Shapley): cada variable rep la seva aportació al risc del disc.

- Ho fem amb el gradient boosting i el random forest.
- Les aportacions sumen la predicció, així que es poden sumar per grups de variables. Per a cada alerta podem dir quina part del risc ve dels comptadors d'error, és a dir, de la regla de Backblaze, i quina part ve d'altres coses. Encaixa amb l'angle propi.
- Gràfics:
  - Un gràfic per franja de temps, per veure si els comptadors pesen menys com més lluny és l'horitzó.
  - L'explicació d'unes quantes alertes reals, per a la part de l'aplicació («per què salta l'avís en aquest disc»).
- Límits:
  - SHAP explica el que ha après el model, no per què fallen els discs.
  - Quan dues variables porten la mateixa informació (el valor d'un comptador i quant ha pujat), es reparteixen el mèrit. Per això ho mirem per grups.

**Mostra d'entrenament.** Hi ha massa files sense fallada per entrenar amb totes.

- Ens quedem amb **totes** les files de discs que fallen i amb una **mostra** de la resta: un 15 % dels discs, triats a sorts de manera fixa, i un de cada deu dimecres.
- Després **corregim les probabilitats** pel fet d'haver tret negatius. Si no ho féssim, el model creuria que les fallades són molt més freqüents del que són.

---

## 6. Decisions

**Preses el 24-11-2026:**

| Decisió | Què hem decidit |
| --- | --- |
| Quins discs entren a cada dimecres (Vikunja #144) | Els vistos en els 7 dies anteriors, amb els seus últims valors. Ara un 95 % de les fallades surten a la franja de 0 a 7 dies d'algun dimecres (abans, un 85 %) |
| Festes (Vikunja #70) | Les etiquetes de desenvolupament es tallen el 16-11-2025 i el test comença el 7-1-2026. L'entrenament no fa servir cap informació posterior al tall, ni tan sols per saber si un disc segueix viu |
| Mètrica per triar models | Detecció amb l'1 % d'alertes. PR-AUC i ROC-AUC com a secundàries |
| Pressupost per a l'aplicació | Dos gràfics: la corba de pressupost (quantes fallades es detecten segons quants discs avisem) i l'estalvi segons el pressupost. L'estalvi compta que cada disc avisat costa 1 i cada fallada no avisada costa ρ (5, 20 o 100 vegades més): estalvi = (ρ − 1) × fallades avisades − falses alarmes. El pressupost òptim és on l'estalvi és màxim |
| Què vol dir «esperança» | Només la supervivència observada amb Kaplan-Meier per grups de risc, fins a 174 dies. No calculem «dies de vida» per disc, per no desviar-nos de la pregunta principal |

**Encara pendents:**

| Decisió | Proposta | Com es decideix |
| --- | --- | --- |
| Quantes dades fem servir per entrenar | Ara, 15 % dels discs sense fallada i un de cada deu dimecres | Amb la corba d'aprenentatge (`learning_curve.py`) al homelab. Entrenar amb tot és viable per al gradient boosting i la logística, però no per a k-NN i SVM |
| Repartiment de la feina | A (dades, validació i aplicació) per a qui té el homelab; B (models i experiment central) per a l'altre | Entre nosaltres |
| Seguiment amb el professor | Ensenyar-li l'angle propi i la definició d'esperança | A la propera classe |

## 7. Següents passos

El calendari és ajustat: entrega el 13-12 i presentació el 14 i el 16-12, amb **assistència obligatòria** tots dos dies.

### Aquesta setmana (23–29 de novembre)

**Abans de res:** executar la corba d'aprenentatge al homelab per decidir quantes dades fem servir, i després la validació creuada (`cv_models.py`).

**Paquet A (dades, validació i aplicació):**

1. **Mostra d'entrenament (Vikunja #34):** aplicar el tall de festes, escollir els discs de la mostra i posar els pesos.
2. **Adaptar `model.py` i `explore.py`** al format nou: un fitxer per dimecres.
3. **Model per franges (#43):** regressió logística i random forest.
4. **Comparació entre franges i horitzons separats (#44):** si les corbes sempre pugen, si ordenen igual de bé i si les probabilitats estan ben calibrades.
5. **Calibratge (#45).**

**Paquet B (models i experiment central):**

1. **Marc d'avaluació (#37):** funcions per calcular la detecció amb un pressupost d'alertes, la PR-AUC, la ROC-AUC i la log loss. Es poden fer ja amb la taula nova, sense esperar la mostra.
2. **Referències R1, R2 i R3 i taxa per model i edat (#38):** surten directament de les columnes, sense entrenar res.
3. **Bootstrap per disc (#39).**
4. **Cronometrar un entrenament de cada model (#28)**, per saber quant trigarà la cerca d'hiperparàmetres.
5. **Cerca d'hiperparàmetres dels set models (#47)**, començant pel gradient boosting, i el notebook `03_models.py` (#49).
6. **Cerca bibliogràfica breu (#40):** comprovar que ningú (ni Backblaze) ha publicat aquesta mateixa anàlisi.

**Fita D2 (29-11):** els models queden **congelats**. A partir d'aquí no s'afegeixen ni variables ni models.

### Setmana del 30 de novembre al 6 de desembre

- **Escala de variables (E1) i discs nets i fallades netes (E2).** És el resultat central.
- **Arbre dibuixat, importància de variables per permutació i SHAP per grups.**
- **Simulació de l'aplicació:** alertes setmanals, classes d'urgència (urgent, programar, vigilar), quants dies d'antelació donem i quin cost té.
- **Esperança per grup de risc:** quant temps sobreviuen realment els discs que el model posa en risc alt, mitjà o baix.
- **Fita D3 (4-12): s'executa el test una sola vegada.**

### Setmana del 7 al 13 de desembre

- Diapositives: 8 minuts, la meitat cadascú.
- Assajos el 9 i l'11, l'últim responent cadascú les preguntes de la part de l'altre.
- Neteja del codi i zip.
- Entrega prevista el 12 (límit, el 13 a les 23:59).

---

## 8. Coses pràctiques

- **Entorn:** Python amb `uv`. S'executa amb `uv run python -I script.py <camins>`. No s'instal·la res amb `pip`.
- **Dades:** no van mai al git. Es passen per fora (carpeta comprimida).
  - Per entrenar només calen `out/landmarks/` (625 MB) i `data/drives.parquet`.
  - El panel sencer (5 GB) només cal per regenerar la taula.
- **Notebooks:** marimo, a `notebooks/`. Llegeixen taules ja processades i no tornen a processar els CSV.
- **Tasques:** a Vikunja, projecte «Aprenentatge-Backblaze». Cada tasca porta l'etiqueta del paquet (A o B).
- **Regles que no es poden trencar:**
  - Separar sempre per disc.
  - No mirar el test fins a la fita D3.
  - No fer servir l'accuracy.
  - No posar a 0 les etiquetes desconegudes.
  - Fer servir llavors aleatòries fixes perquè tots dos obtinguem el mateix.
