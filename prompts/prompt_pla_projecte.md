# Prompt: pla del projecte (per a un xat nou)

Ets el company tècnic d'una parella d'estudiants de 3r de Matemàtica Computacional i Analítica de Dades (MatCAD) que fan el treball d'Aprenentatge Computacional. La teva feina en aquest xat és **dissenyar el pla del projecte**, no implementar-lo. **No escriguis scripts, notebooks ni codi d'experiments.** Els dos membres de la parella decidiran plegats què s'implementa a partir del teu pla. Pots llegir fitxers i dades i fer consultes ràpides per comprovar fets (mides, recomptes, columnes), però el lliurable és un document.

Respon en català.

## 1. Context que has de llegir primer

El directori de treball és el repositori del projecte (`projecte/`). Llegeix, en aquest ordre:

1. `AGENTS.md`: convencions, regles de dades i metodologia que no s'han de trencar.
2. `enunciat_treball.md`: l'enunciat del professor. Fixa't que l'avaluació es basa en una **presentació de 8 minuts**: la qualitat de la pregunta, el disseny dels experiments, els resultats i la capacitat de respondre preguntes dels companys.
3. `dataset/estudi_backblaze.md`: estudi de viabilitat amb el trimestre Q4 2025. Inclou la construcció de l'objectiu, els primers resultats de sis models, la prova sense comptadors d'error, la comparació amb Kaplan-Meier i els riscos.
4. `dataset/backblaze/context_processament.md`: com estan processades les dades, què fa cada script, la proposta de separació entre train i test i la llista de pendents.
5. `dataset/backblaze/scripts/*.py`: llegeix-los per saber exactament què fan ara (consolidació a Parquet, landmarks, models).
6. `dataset/backblaze/out/*.csv` i `*.txt`: resultats actuals.
7. **Teoria de l'assignatura**, al directori pare (`../`):
   - `../formulari.md`: resum dels temes amb la notació del curs. Llegeix-lo sencer.
   - `../teoria/*.pdf`: diapositives de Statistical Learning, Linear Models, kNN, Naive Bayes, SVM i Decision Trees and Random Forests. Consulta-les per justificar decisions amb el vocabulari i els resultats que s'han vist a classe. No calen totes senceres, però sí les parts rellevants per a cada decisió.
   - No llegeixis `../seguiment.md` ni `../examens/`: són d'estudi personal i no formen part del projecte.
   - Si alguna d'aquestes rutes no existeix al teu entorn, digues-ho i continua amb la resta.

## 2. Situació i restriccions

- **Dataset:** Backblaze Drive Stats, amb una fila per disc i dia: model, capacitat, SMART raw i `failure`. Hi ha uns 340.000 discs i unes 950 fallades per trimestre.
- **Dades ja processades:** Q4 2025, consolidat a `data/panel/2025Q4.parquet` (31 M de files, 889 MB) i `data/drives.parquet`. Un trimestre ocupa 12 GB en CSV.
- **Requisits del professor, acordats a classe**, a més de l'enunciat:
  - L'objectiu es construeix fixant un horitzó temporal t (el disc falla abans de t).
  - Cal fer diversos models per horitzó o per franja de temps, per obtenir una probabilitat acumulada.
  - Cal posar el focus en l'aplicació del model.
  - Els creadors del dataset no poden haver publicat la mateixa anàlisi.
- **Reserva central:** Backblaze decideix en part que un disc ha fallat a partir de SMART 5, 187, 188, 197 i 198, i el model també els fa servir. Sense aquests comptadors, la PR-AUC cau unes 10 vegades. Això pot ser un problema o el cor de la pregunta de recerca.
- **Calendari:** avui és 12 de novembre de 2026. El lliurament (PDF de la presentació i zip del codi) és el **13 de desembre a les 23:59**, i les presentacions, el 14 i el 16 de desembre. Són dues persones durant unes 4 setmanes, compaginant-ho amb altres assignatures.
- **Màquina:** 4 nuclis, 15 GB de RAM i uns 120 GB de disc lliure. El company pot tenir una màquina diferent. Res del pla pot exigir carregar el panel complet en memòria.
- **Eines:** Python amb uv. Els scripts pesats són de línia d'ordres, i l'anàlisi i les figures es faran amb notebooks de **marimo**, perquè el professor el fa servir.
- **Ambició:** volem fer alguna cosa **innovadora o, com a mínim, interessant** dins del marc acordat amb el professor. L'enunciat diu explícitament que l'objectiu és explorar, comparar models i arribar a una conclusió ben argumentada, no aconseguir el millor rendiment absolut. Prioritza les idees que generin una història clara per a 8 minuts i que es puguin defensar davant de preguntes.

## 3. Què ha de contenir el pla

Organitza'l en aquestes parts. Per a cada decisió important, dona **la teva recomanació**, l'alternativa principal, el cost (temps i còmput) i què es perd si no es fa. Distingeix entre el que és imprescindible i el que és opcional si sobra temps.

### Part 0. Pregunta de recerca

L'enunciat avalua sobretot la pregunta. Proposa 2 o 3 preguntes concretes i responibles amb experiments, i recomana'n una com a eix. Algunes direccions possibles, sense limitar-te a aquestes:

- Quant del senyal predictiu ve de la mateixa definició de fallada (fuita) i quant és predicció real.
- Fins a quina antelació es pot detectar una fallada amb un pressupost d'alertes realista.
- Si un model entrenat amb trimestres antics aguanta en trimestres nous (canvis en el parc de discs).
- Com es comparen els models per horitzó amb un model únic que doni tota la corba de risc.

La resta del pla ha d'estar al servei de la pregunta triada.

### Part 1. Processament i estructura de les dades

- **Diversos trimestres:**
  - Quants i quins. Comprova a la web de Backblaze quins hi ha publicats fins avui; si no ho pots comprovar, digues-ho.
  - Què hi guanyem: més fallades, horitzons complets, test temporal, estudi de deriva.
  - Cost en disc i temps de processament.
  - Problemes en unir-los: canvis de columnes SMART entre trimestres, models nous o retirats, discs que surten i tornen, números de sèrie.
- **Estructura de les dades:** capes (CSV, panel diari per trimestre, taula de discs, taula de landmarks o de períodes) i quin format i granularitat ha de tenir cada capa perquè els experiments siguin barats.
- **Disseny dels landmarks:**
  - Freqüència (setmanal o cada 2 setmanes) i correlació entre files del mateix disc.
  - Quines variables temporals construir: deltes a 1, 7 i 30 dies, màxims, si el valor ha estat mai superior a 0, edat, hores d'ús.
  - Com tractar els SMART que només reporten alguns fabricants, i si cal normalitzar per fabricant o per model.
- **Formulacions de l'objectiu**, comparades i amb una recomanació:
  - (a) Classificació binària per a cada horitzó fix y_t, t = 7, 30 i 60, que és el que hi ha ara.
  - (b) **Temps discret per franges**: una fila per disc, landmark i franja (0–7, 7–30, 30–60 dies), amb el risc condicionat de fallar en cada franja (hazard discret). La probabilitat acumulada surt del producte, i això garanteix corbes monòtones. Ara mateix, amb models separats, el 27 % de les corbes del random forest baixen.
  - (c) **Horitzó com a variable d'entrada**: el model rep (x, t) i dona P(T ≤ t | x), de manera que una sola funció dona tota la corba.
  - (d) Censura: com tractar-la en cada formulació.
- **Partició i mostreig:**
  - Concreta la proposta de `context_processament.md`: test separat en el temps, separació de 60 dies, `GroupKFold` per disc dins del període de desenvolupament, i potser una avaluació extra només amb discs nous.
  - Mostreig dels discs sense fallada i correcció de les probabilitats.
  - Mida de les taules resultants i si caben a la màquina.

### Part 2. Models candidats i com combinar-los

- **Base:** els models del curs (regressió logística amb Ridge o Lasso, k-NN, Naive Bayes, SVM, arbres i random forest) i referències: atzar, regla de Backblaze, Kaplan-Meier per model de disc o per edat.
- **Adaptacions de cada model al problema.** Per exemple:
  - Naive Bayes falla amb comptadors quasi sempre a 0. Es podria provar amb SMART binaritzats o categòrics, o amb una transformació.
  - k-NN i SVM depenen de l'escala i del cost en files.
  - La regressió logística és la forma natural del model de franges (b).
- **Combinacions i idees innovadores dins del marc de l'assignatura.** No tinguis por de proposar, però justifica-ho amb la teoria. Algunes idees:
  - Model de franges amb regressió logística sobre dades per període.
  - Model únic amb l'horitzó com a variable.
  - Ensembles o stacking de models del curs.
  - Model en dues etapes: regla, i després model per als discs que no la disparen.
  - Models per fabricant o un model global.
  - Model "sense fuita" (sense comptadors d'error) i model complet, comparats com a experiment central.
  - Si proposes algun model fora del temari, digues per què val la pena i com el connectaríem amb el que s'ha vist.
- Quins models arriben a la presentació final i quins es queden en la fase d'exploració.

### Part 3. Optimització d'hiperparàmetres i compromís realista

- **Procediment:**
  - Validació creuada dins del període de desenvolupament, agrupada per disc, i test temporal intocable.
  - Espais de cerca petits i justificats.
  - Recompte de models entrenats (valors × grups + 1) i pressupost de còmput.
- **Mètrica que reflecteixi el problema real.** Pensa en l'operador d'un centre de dades:
  - Té un pressupost d'alertes (quants discs pot revisar o substituir per setmana).
  - Una fallada no prevista costa una reconstrucció RAID, risc de perdre dades i temps d'enginyeria.
  - Una falsa alarma costa substituir o migrar un disc sa.
  - Necessita prou antelació per actuar (temps per demanar recanvis o migrar dades).
  - Proposa com traduir-ho en mètriques i llindars: fallades detectades amb k alertes, precisió de les alertes, cost esperat amb una ràtio de costos (i sensibilitat a aquesta ràtio), i antelació mitjana de la detecció.
- **Calibratge:** quan importa (si les probabilitats s'han de fer servir com a tals) i com fer-lo (Platt o regressió isotònica) sense contaminar el test.
- **Estabilitat:** variabilitat entre grups i entre trimestres, i intervals de confiança amb bootstrap per disc.

### Part 4. Relació amb la teoria i justificació de les decisions

- Una taula que relacioni cada decisió del pla amb el concepte del curs que la justifica, citant el tema o la diapositiva. Per exemple:
  - Log loss i màxima versemblança per a la regressió logística i el model de franges.
  - Biaix i variància en triar k, la profunditat de l'arbre, C i γ, o la regularització.
  - Corbes ROC i PR amb classes desequilibrades.
  - Fuites d'informació a la validació.
  - Maledicció de la dimensionalitat al k-NN.
  - Hipòtesi naive amb SMART correlacionats.
  - Importància per impuresa i per permutació al random forest.
- Les **preguntes probables dels companys o del professor** durant la presentació, amb una resposta breu per a cadascuna. Per exemple: per què no fer servir l'accuracy, per què no barrejar files del mateix disc, si el model no aprèn simplement la regla de Backblaze, o per què una precisió del 8 % és útil.

### Part 5. Pla de treball i presentació

- Fases amb dates fins al 13 de desembre, amb punts de decisió (per exemple: si al final de la setmana 2 X no funciona, passem a Y).
- Repartiment de la feina entre els dos membres, de manera que cadascú pugui presentar la meitat amb coneixement de causa.
- **Esquema de la presentació de 8 minuts:** quines 6 a 8 diapositives, quina figura clau a cadascuna i quin missatge.
- Quins notebooks de marimo caldrien al final, només el nom i el propòsit, sense codi.
- Riscos del pla i com mitigar-los.

## 4. Format del lliurable

- Escriu el pla a `pla_projecte.md`, a l'arrel del repositori. Fes-lo en català, amb seccions que corresponguin a les parts 0–5 i taules quan ajudin a comparar opcions.
- Comença amb un **resum d'una pàgina**: pregunta recomanada, formulació de l'objectiu, models, mètrica principal i calendari.
- Acaba amb una llista de **decisions obertes perquè la parella les discuteixi**, cadascuna amb la teva recomanació.
- Marca clarament què has **comprovat** (llegint dades, scripts o webs) i què és una **suposició**.
- No modifiquis cap altre fitxer del repositori i no facis commits.

Quan acabis, resumeix en el xat les 3 o 4 decisions més importants i les preguntes que necessites que et responguem.
