# Pla del projecte: predicció de fallades de discs (Backblaze)

Versió del 12-11-2026, actualitzada amb les respostes de la parella (pregunta enviada al professor, temari, homelab i repartiment pendent) i, el 18-11-2026, amb el tema nou de gradient boosting. El 19-11-2026 s'hi incorporen les decisions de la parella (taula final) i l'anàlisi preliminar de Nadal (1.5 bis). Document de disseny per discutir entre els dos membres. No conté codi ni substitueix `AGENTS.md`, `estudi_backblaze.md` ni `context_processament.md`.

**Convenció.** **[C]** vol dir comprovat avui llegint dades, scripts, diapositives o la web de Backblaze. **[S]** vol dir suposició o estimació que cal confirmar. Les xifres sense etiqueta provenen de l'estudi de viabilitat. L'annex A recull totes les comprovacions.

---

## Resum (una pàgina)

**Pregunta (la que vau enviar al professor; no es canvia).** *A partir de l'estat SMART d'un disc en un moment donat, quina probabilitat hi ha que falli en els propers t dies, quina esperança podem donar a aquests discs i com podem fer servir aquesta informació per saber quins discs s'han de substituir i quan?*

Té tres parts, i el pla hi afegeix un angle propi:

| Part | Com es respon |
| --- | --- |
| **Probabilitat** de fallar en t dies | Model per franges de temps (0–7, 7–30, 30–60 dies) que dona una probabilitat acumulada monòtona i calibrada |
| **Esperança** | Supervivència observada dels discs del test agrupats pel risc predit, seguits fins a 174 dies; per disc, dies de vida esperats dins dels 60 dies |
| **Quins substituir i quan** | Alertes setmanals amb pressupost; classe d'urgència segons la franja on es concentra el risc; antelació i cost |
| **Angle propi: de què ens podem refiar** | La pregunta diu «a partir de l'estat SMART», i Backblaze declara fallades en part pels mateixos SMART. Mesurem quina part del senyal és aquesta regla i quina és anticipació |

L'angle propi es respon amb tres experiments:

1. **Escala de variables per franja de temps.** Entrenem el mateix model amb grups de variables cada vegada més rics (context, SMART que no són d'error, senyal operatiu, nivells dels comptadors d'error, canvis dels comptadors) i mirem quant guanya cada grup a cada franja.
2. **Discs nets i discs marcats.** Avaluem per separat els discs que a la data de referència tenen els cinc comptadors a zero (92 % del parc i un 35 % de les fallades a 30 dies **[C]**) i els que ja en tenen algun per sobre de zero.
3. **Fallades netes i brutes.** Separem les fallades segons si el disc tenia algun comptador per sobre de zero l'últim dia (73 %) o cap (27 %) **[C]**. Les segones no poden venir de la regla SMART.

**Formulació de l'objectiu.** Temps discret per franges: un model per franja que estima el risc de fallar en aquella franja si el disc hi ha arribat viu. La probabilitat acumulada a 7, 30 i 60 dies surt del producte i és monòtona per construcció. Els models separats per horitzó (el que hi ha ara) queden com a comparació.

**Dades.** Sis trimestres, del Q1 2025 al Q2 2026 (publicats tots **[C]**). Dates de referència setmanals. Desenvolupament: del 5-2-2025 al 5-11-2025 (40 dates). Separació de 63 dies. Test: del 7-1-2026 al 29-4-2026 (17 dates), que no es toca fins al final. Aquestes dates són la proposta de partida: es fixen després de l'anàlisi preliminar de Nadal (1.5 bis).

**Models.** Només el temari: els set models del curs (ara també gradient boosting) amb una cerca petita d'hiperparàmetres. A la presentació arriben la regressió logística, el random forest i el gradient boosting, més tres referències sense aprenentatge: la regla de Backblaze, una regla de canvis en 7 dies i la taxa de fallada per model i edat. L'arbre poc profund es mostra dibuixat per comparar-lo amb la regla.

**Mètrica principal.** Percentatge de fallades detectades amb un pressupost fix d'alertes per setmana (1 % del parc per triar models; corba sencera i 0,1 % per a l'aplicació), amb intervals de confiança per bootstrap per disc. PR-AUC i ROC-AUC com a secundàries. Per a l'aplicació, antelació de l'avís i cost segons la ràtio entre fallada imprevista i falsa alarma.

**On s'executa.** Processament i entrenament al homelab, amb CPU. Els notebooks només llegeixen taules petites i funcionen a qualsevol portàtil. No es fa servir la GPU (secció 1.6).

**Repartiment.** Encara no decidit. El pla descriu dos paquets de feina equilibrats (A i B) que es poden assignar quan l'abast estigui acordat.

**Calendari.**

| Setmana | Dates | Fita |
| --- | --- | --- |
| 0 | 12–15 nov. | Acordar aquest pla, baixar i consolidar 5 trimestres |
| 1 | 16–22 nov. | Taula de dates de referència nova, partició, referències i mètriques |
| 2 | 23–29 nov. | Models i hiperparàmetres amb validació creuada; es congelen el 29 |
| 3 | 30 nov.–6 des. | Experiments centrals, execució única del test, figures |
| 4 | 7–13 des. | Diapositives, assajos, zip del codi. Entrega prevista el 12, límit el 13 |

---

## Troballes d'avui que condicionen el pla

Totes **[C]**; el detall és a l'annex A.

1. **Les fallades cauen per Nadal, no a final de trimestre.** Al Q4 2025 les fallades setmanals passen d'unes 100 a l'octubre a 33, 10 i 6 les tres últimes setmanes de desembre. Al Q1 2026 no hi ha cap caiguda al final (73 i 74 les dues últimes setmanes senceres) i el gener arrenca baix (12 en quatre dies, després 66). És un efecte de calendari: l'etiqueta depèn de quan algú retira el disc. Conseqüència: el test no ha de ser novembre–desembre de 2025, com proposava `context_processament.md`.
2. **Sortir de les dades no és neutre.** Dels 1.904 discs que surten del Q4 2025 sense fallar, 1.627 (85 %) tornen al Q1 2026, i 87 d'aquests fallen aquell trimestre (5,3 %, contra el 0,30 % del parc). A més, el 27 % dels discs que fallen tenen dies absents dins del trimestre, contra el 3 % del total. Conseqüències: amb diversos trimestres gairebé tota la «censura» desapareix, i els dies d'absència són una variable candidata forta que no és un comptador SMART.
3. **Una regla senzilla ja fa molta feina.** A la primera data de referència, «algun comptador d'error ha pujat en 7 dies» marca el 0,72 % dels discs i conté el 35 % de les fallades a 30 dies (precisió 5,6 %). La regressió logística de l'estudi, amb l'1 %, en detecta el 41 %, i el random forest, el 51 %. La referència «algun comptador > 0» de l'estudi (8 % amb l'1 % d'alertes) és massa feble: cal comparar amb regles millors.
4. **El senyal dels comptadors ve de lluny.** El 55 % dels discs que fallen ja tenen algun comptador per sobre de zero 60 dies abans, el 63 % a 30 dies i el 73 % l'últim dia. No és evident que el senyal «de la regla» es concentri a la primera franja; és una hipòtesi oberta.
5. **`consolidate.py` necessita dos canvis abans d'unir trimestres** (secció 1.1): la llista de columnes SMART es decideix per trimestre, i la taula de discs carrega tots els trimestres a memòria.

---

## Part 0. Pregunta de recerca

### La pregunta enviada i com es respon

> *A partir de l'estat SMART d'un disc en un moment donat, quina probabilitat hi ha que falli en els propers t dies, quina esperança podem donar a aquests discs i com podem fer servir aquesta informació per saber quins discs s'han de substituir i quan?*

La pregunta ja està formalitzada amb el professor, així que el pla la respon tal com és. Cada part necessita una definició operativa:

| | Part de la pregunta | Definició operativa | Experiments | Què es presenta |
| --- | --- | --- | --- | --- |
| **Q1** | «Quina probabilitat hi ha que falli en els propers t dies» | P(T ≤ t \| estat a la data de referència) per a t = 7, 30 i 60, amb el model de franges, calibrada | E3, E4, calibratge | Taula de models per franja, corbes acumulades, fiabilitat |
| **Q2** | «Quina esperança podem donar a aquests discs» | Per grup de risc: supervivència observada al test fins a 174 dies. Per disc: dies de vida esperats dins dels 60 dies | E7 | Corbes de supervivència per grup de risc |
| **Q3** | «Quins discs s'han de substituir i quan» | Quins: els k primers per risc, o els que superen el llindar de cost. Quan: classe d'urgència segons la franja | E5 | Corba de pressupost, classes d'urgència, antelació |
| **Q0** | «A partir de l'estat SMART» (angle propi) | Quina part de l'estat SMART porta la informació, i si és circular amb la definició de fallada | E1, E2 | Resultat central |

**Per què cal Q0.** Sense Q0, les respostes a Q1–Q3 es poden llegir malament: una probabilitat alta pot voler dir «Backblaze retirarà aquest disc perquè ja té errors», no «aquest disc es deteriorarà». Q0 diu fins a quin punt cada cosa és certa, i és el que fa el treball diferent d'un classificador més.

**Altres preguntes considerades**, que queden com a subproductes:

- *Un model entrenat el 2025 aguanta el 2026?* Surt gratis de comparar la validació creuada amb el test temporal. Es presenta en una frase.
- *Franges o models separats per horitzó?* És una decisió de mètode dins de Q1 (experiment E4).

### Formulació operativa de l'angle propi (Q0)

- **Q0a.** Quants punts de detecció amb l'1 % d'alertes aporta cada grup de variables a cada franja.
- **Q0b.** Entre els discs nets a la data de referència, el model supera la taxa de fallada per model i edat? Aquí el model no pot estar llegint l'estat actual de la regla.
- **Q0c.** Quina part de les fallades netes (cap comptador > 0 l'últim dia) es detecta, comparada amb les brutes.
- **Q0d.** Amb un pressupost realista, quantes fallades s'avisen amb almenys 7 dies de marge.

**Què es conclou segons el resultat:**

| Resultat | Conclusió |
| --- | --- |
| Sense comptadors d'error no queda gairebé res, i en discs nets el model no supera la taxa per model i edat | El model és la regla de Backblaze amb millor ordenació. Serveix per prioritzar entre discs ja marcats, no per anticipar |
| Queda senyal en discs nets o en fallades netes | Hi ha anticipació real; es quantifica i es diu d'on ve (càrrega, absències, edat) |
| El guany dels comptadors cau de la franja 0–7 a la 30–60 | L'avís útil és de curt termini; l'aplicació realista és la revisió setmanal |
| El guany es manté a 30–60 dies | Els comptadors avisen amb setmanes de marge; la regla de Backblaze és tardana, no errònia |

**Límit que cal dir en veu alta.** Les dades no diuen per què es va retirar cada disc. «Net» i «brut» són aproximacions a «reactiva» i «proactiva». L'efecte de Nadal mostra que l'etiqueta també depèn de l'operativa del centre de dades.

### Què vol dir «esperança» (Q2)

Amb horitzons de fins a 60 dies i un 0,17 % de fallades, el model no pot donar una esperança de vida completa sense extrapolar. Opcions:

| Opció | Què és | A favor | En contra | Recomanació |
| --- | --- | --- | --- | --- |
| **Supervivència observada per grup de risc** | Agrupar els discs del test pel risc predit a la primera data (7-1-2026) i seguir-los amb Kaplan-Meier fins al 30-6-2026 (174 dies) | Sense supòsits. Figura clara. Kaplan-Meier ja el teniu. Comprova el model més enllà dels 60 dies | És per grup, no per disc. Seguiment màxim de 174 dies | **Principal** |
| Esperança restringida a 60 dies, per disc | Àrea sota la corba de supervivència del model de franges fins a 60 dies | Surt del model sense cap supòsit | Per a gairebé tots els discs val 59,9 dies; només és expressiva en discs de risc alt | Calcular-la i ensenyar-la per a discs de risc alt |
| Extrapolació amb risc constant | Vida mediana = ln 2 / λ, amb λ el risc diari deduït de la probabilitat a 60 dies | Dona un número per disc | Suposa que el risc no canvia. Per a un disc sa dona desenes d'anys | Només com a frase per a discs de risc alt, contrastada amb la supervivència observada |
| Taula de vida per edat i fabricant | Risc per any d'edat (hores d'ús), a partir de `drives.parquet` | Barata. És la referència «taxa per model i edat» | No fa servir l'estat SMART. Backblaze ja publica anàlisis de vida per edat **[S]**, i el professor no vol repetir anàlisis dels creadors | Només com a referència, no com a resultat |

- **Grups de risc proposats:** el 0,1 % amb més risc, del 0,1 % a l'1 %, de l'1 % al 10 % i la resta.
- **Què es diu a la presentació:** per a cada grup, quin percentatge segueix viu a 60, 120 i 174 dies, i la vida mediana si el grup hi arriba.
- **Cost:** baix. Només calen les prediccions de la primera data de test i `drives.parquet`. Forma part de l'execució única del test; abans es pot assajar amb prediccions de validació creuada al desenvolupament.

### Què vol dir «quan» (Q3)

El model de franges permet assignar a cada disc una classe d'urgència amb les probabilitats calibrades:

| Classe | Condició | Acció |
| --- | --- | --- |
| Urgent | P(T ≤ 7) per sobre del llindar | Copiar dades o substituir aquesta setmana |
| Programar | P(T ≤ 30) per sobre del llindar, i no urgent | Demanar el recanvi i substituir a la propera finestra |
| Vigilar | P(T ≤ 60) per sobre del llindar, i cap de les anteriors | Revisar cada setmana |
| Cap acció | La resta | — |

- **Llindar:** el que resulti del pressupost d'alertes, o 1/ρ si es treballa amb costos (secció 3.2).
- **Com s'avalua:** per a cada classe, quants discs fallen realment dins de la seva franja, i a quina classe queden les fallades del test.

---

## Part 1. Processament i estructura de les dades

### 1.1 Diversos trimestres

**Publicats avui [C]:** tots els trimestres fins al Q2 2026. Des de 2016 són trimestrals.

| Trimestre | Zip | CSV | Fitxers | Notes |
| --- | --- | --- | --- | --- |
| Q1 2025 | 1,02 GB | 10,9 GB | 90 | |
| Q2 2025 | 1,07 GB | 11,3 GB | 91 | |
| Q3 2025 | 1,1 GB | 11,3–11,7 GB | 91 | La web dona dues mides |
| Q4 2025 | 1,2 GB | 12,2 GB | 92 | Ja consolidat: 945 fallades |
| Q1 2026 | 1,3 GB | 12,0 GB | 90 | 1.030 fallades, 30,6 M files **[C]** (la web diu 91 fitxers; n'hi ha 90) |
| Q2 2026 | 2,1 GB | 11,7 GB | 91 | 1.498 fallades i 354.415 discs segons la web |

**Recomanació: sis trimestres, Q1 2025 – Q2 2026.**

| Opció | Què dona | Cost | Què es perd |
| --- | --- | --- | --- |
| **Sis trimestres (recomanat)** | Unes 7.000 fallades **[S]**, 40 dates de desenvolupament, 17 de test, test lluny de Nadal | 5 descàrregues; uns 45 min en total **[S]**; 5,5 GB finals | — |
| Quatre trimestres (Q3 2025 – Q2 2026), mínim viable | El mateix test, 14 dates de desenvolupament | 3 descàrregues | Menys fallades per entrenar, sobretot a la franja 0–7 |
| Afegir 2024 | Estudi de deriva a dos anys | 4 descàrregues més, 3,6 GB | Res: la deriva no és part de la pregunta. No ho faria |
| Un o dos trimestres | — | — | No hi ha lloc per a desenvolupament, separació i test |

**Cost mesurat o estimat:**

- Descàrrega: 1,3 GB en 78 s en aquesta màquina **[C]**.
- Consolidació: uns 200 s per trimestre **[C]** (`consolidate_log.txt`).
- Disc: processant un trimestre cada vegada i esborrant els CSV, el pic és d'uns 14 GB. Al final, 6 panels d'uns 0,9 GB. Els CSV del Q4 (`data/q4`, 12 GB) ja es poden esborrar.
- Millora opcional: llegir els CSV directament del zip sense descomprimir. Avui s'han llegit així 4 columnes dels 90 dies del Q1 2026 en pocs minuts **[C]**. Estalvia els 12 GB temporals.

**Problemes en unir trimestres i què fer-hi:**

| Problema | Estat | Proposta |
| --- | --- | --- |
| `consolidate.py` conserva les columnes SMART amb menys del 99,9 % d'absents **de cada trimestre**. Dos trimestres poden quedar amb columnes diferents, i `pyarrow.dataset` agafa per defecte l'esquema del primer fitxer | **[C]** llegint el codi; no executat amb dos trimestres | Fixar la llista de columnes (les 63 del Q4 2025, a `2025Q4_columns.csv`) i aplicar-la a tots |
| La taula de discs llegeix 7 columnes de **tots** els trimestres alhora i les ordena. Amb 186 M de files i `model` com a text, estimo més de 10 GB | **[C]** el codi; **[S]** la memòria | Agregar per trimestre i combinar després les agregacions (mínim de `first_seen`, màxim de `last_seen`, primera fallada) |
| `build.py` llegeix tota la columna `date` per saber quins dies hi ha | **[C]** | Amb 186 M de files són uns 750 MB; acceptable, però es pot treure de les metadades dels grups de files |
| Columnes SMART noves entre trimestres | Q4 2025 i Q1 2026 tenen exactament les mateixes 93 columnes `_raw` i les mateixes 11 columnes base **[C]**. Les columnes base no canvien des del Q3 2023 **[C]**. Q1–Q3 2025 i Q2 2026 no comprovats | El pas 1 de `consolidate.py` ja fa la unió per capçalera. Mirar el recompte de columnes de cada trimestre en consolidar |
| Models de disc nous o retirats | Al Q1 2026 entren 10.653 números de sèrie nous **[C]** | Definir la llista de models freqüents **només amb el desenvolupament**. La resta va a «altres», amb fabricant i capacitat com a suport |
| Discs que surten i tornen | 85 % dels que surten tornen **[C]** | L'etiqueta es calcula amb la data de fallada i l'últim dia vist **en tot el període**, no per trimestre. `build.py` ja ho fa si `drives.parquet` cobreix tots els trimestres |
| Discs que fallen i reapareixen | 1 de 945 **[C]** | Mantenir la regla actual: després de la primera fallada, el disc s'exclou |
| Dies absents | 10.641 discs amb forats al Q4, mediana de 2 dies **[C]** | Per als canvis a 7 i 30 dies, fer servir l'últim valor disponible dins d'una finestra curta i guardar els dies d'absència com a variable |

### 1.2 Capes de dades

| Capa | Fitxer | Granularitat | Mida | Qui la llegeix |
| --- | --- | --- | --- | --- |
| 0. Original | zip per trimestre | fila per disc i dia, CSV | 1–2 GB | Només `consolidate.py`; es pot esborrar |
| 1. Panel | `data/panel/<trimestre>.parquet` | fila per disc i dia, un grup de files per dia | 0,9 GB per trimestre **[C]** | Només `build.py`, sempre filtrant per data i columnes |
| 2. Discs | `data/drives.parquet` | fila per disc | uns 5 MB | Tot |
| 2b. Presència | `data/presence.parquet` (nova) | fila per disc amb els dies que hi és, o matriu de bits | menys de 30 MB **[S]** | `build.py`, per als dies d'absència |
| 3. Dates de referència | `out/landmarks/L=<data>.parquet` (un fitxer per data) | fila per disc i data, amb variables, `t_obs`, `event`, partició | 19 M de files, uns 0,7 GB **[S]** | Scripts d'entrenament i avaluació, data a data |
| 4. Mostra d'entrenament | `out/train_sample.parquet` | totes les files amb fallada a 60 dies i una mostra de la resta, amb pes | unes 230.000 files, menys de 50 MB **[S]** | Models i notebooks; cap en memòria de sobres |
| 5. Prediccions | `out/pred_<experiment>.parquet` | fila per disc i data, amb etiquetes i una columna per model i franja | 6 M de files al test | Notebooks: mètriques, bootstrap, figures |

L'estimació de la capa 3 surt d'escalar el fitxer actual (35,9 MB per 1,34 M de files **[C]**).

**Per què així.** La capa 3 no es carrega mai sencera: l'entrenament fa servir la capa 4 i l'avaluació recorre la capa 3 data a data i només guarda prediccions. La capa 5 permet que el bootstrap, el calibratge i totes les figures es facin als notebooks sense tornar a entrenar. Les etiquetes de franja no es desen com a taula llarga: es deriven de `t_obs` i `event` en llegir.

### 1.3 Dates de referència i variables

**Freqüència: setmanal** (dimecres, com ara).

| Opció | A favor | En contra |
| --- | --- | --- |
| **Setmanal (recomanat)** | És la cadència real de l'operador. Cada fallada cau a la franja 0–7 d'exactament una data | Files d'un mateix disc molt correlacionades |
| Cada 2 setmanes | La meitat de files | La meitat de les fallades no apareixen mai a la franja 0–7, que ja és la més escassa (323 contra 1.422 a 30 dies) |

La correlació no es resol reduint files, sinó amb validació agrupada per disc i bootstrap per disc. Als discs sense fallada sí que es pot aprimar: amb una de cada deu dates n'hi ha prou.

**Variables, organitzades en els grups de l'experiment central:**

| Grup | Variables | Comentari |
| --- | --- | --- |
| **G0 Context** | fabricant, model (els més freqüents al desenvolupament), capacitat, edat (SMART 9 en anys) | El que se sap sense mirar la salut del disc |
| **G1 SMART que no són d'error** | 1, 7, 10, 12, 190/194, 192, 193, 199, 240, 241, 242; canvi a 30 dies de 241 i 242 (càrrega recent) | SMART 1 i 7 tenen significat diferent segons el fabricant |
| **GO Senyal operatiu** | dies absent en els últims 30 dies, dies des que va tornar, dies des de la primera aparició | Nou. Potent segons la troballa 2, però pot reflectir que Backblaze ja sospitava del disc. Per això va en un grup a part |
| **G2 Nivell dels comptadors d'error** | 5, 187, 188, 197, 198 amb `log1p`, i indicador de > 0 | La regla de Backblaze |
| **G3 Dinàmica dels comptadors** | canvi a 1, 7 i 30 dies; indicador «ha pujat» | El 53 % dels discs que fallen tenen algun comptador que puja l'última setmana **[C]** |

- **Cost dels canvis a 1, 7 i 30 dies:** quatre lectures d'un dia per data de referència. Amb unes 65 dates són unes 260 lectures; minuts **[S]**.
- **Màxims a 30 dies:** no els faria. Quatre dels cinc comptadors són acumulatius i el màxim és el valor actual. Només SMART 197 pot baixar; n'hi ha prou amb el canvi a 30 dies.
- **«Ha estat mai > 0»:** per a comptadors acumulatius és el mateix que «és > 0 ara».

**SMART que només reporten alguns fabricants.** SMART 187 i 188 falten a l'1,7 % dels discs Seagate i a gairebé el 100 % de la resta **[C]**.

| Opció | Per a quins models | Recomanació |
| --- | --- | --- |
| Indicador d'absent i imputació per la mediana del desenvolupament (ara) | Tots | **Base.** Per als arbres n'hi ha prou |
| Deixar els absents com a NaN | Gradient boosting | **Recomanat per a aquest model.** `HistGradientBoostingClassifier` aprèn a cada partició cap a quin costat envia els absents **[C]**. Com que falten per fabricant, és exactament la informació que volem |
| Estandarditzar dins de cada fabricant (mitjana i desviació del desenvolupament) | Logística, k-NN, SVM | **Provar-ho** per a SMART 1 i 7; és un canvi petit |
| Un model per fabricant | Logística i random forest | Opcional. Amb sis trimestres hi ha prou fallades per a Seagate, Toshiba i HGST; WDC queda just (95 per trimestre **[C]**) |

### 1.4 Formulació de l'objectiu

Notació: T és el nombre de dies fins a la fallada des de la data de referència L. Franges B1 = (0, 7], B2 = (7, 30], B3 = (30, 60].

| | (a) Binari per horitzó | (b) Franges (temps discret) | (c) Horitzó com a entrada |
| --- | --- | --- | --- |
| Què estima | P(T ≤ t) amb un model per a cada t | h_k = P(T ∈ B_k \| T > inici de B_k), un model per franja | Una sola funció f(x, t) = P(T ≤ t) |
| Corba acumulada | Pot baixar: 5 % (logística) i 27 % (random forest) | 1 − ∏(1 − h_k); **monòtona sempre** | Només és monòtona si es restringeix el model |
| Censura | S'exclou la fila de l'horitzó t | El disc compta a les franges que ha completat i s'exclou de la resta | S'exclouen les parelles (x, t) desconegudes |
| Què aprèn cada model | Barreja fallades imminents i llunyanes | Cada franja per separat: mesura l'antelació | Comparteix paràmetres entre horitzons |
| Encaix amb el professor | «Diversos models per horitzó» | «Per franja de temps», «probabilitat acumulada» | Un sol model |
| Encaix amb el curs | Classificació binària | Producte de Bernoullis: la log loss és la versemblança | Classificació binària amb files repetides |
| Cost | Ja fet | Tres taules filtrades de la mateixa mostra | Triplica les files i les correlaciona |

**Recomanació: (b) com a formulació principal i (a) com a comparació.**

- **(b1) Un model per franja**, amb qualsevol classificador del curs. És la versió que es presenta.
- **(b2) Una sola regressió logística** sobre les files de les tres franges apilades, amb la franja com a variable i algunes interaccions. És el model clàssic de supervivència en temps discret. Opcional: és elegant i barat, però no cal per a la història.
- **(c)** no la faria. Es pot esmentar a les preguntes com a alternativa.

**Què es perd si no es fa (b):** les corbes monòtones i, sobretot, la mesura de l'antelació. Amb (a), `y_60` barreja la fallada de demà amb la de d'aquí a 55 dies.

**Censura (d).** Amb sis trimestres, un disc absent uns dies que després torna té etiqueta coneguda (no va fallar). Només queden censurats els discs que marxen i no tornen (uns 277 dels 1.904 del Q4 **[C]**) i les últimes dates abans del final de les dades, que ja excloem. Amb (b) no cal Kaplan-Meier per corregir res; es manté com a comprovació del calibratge mitjà, com a l'estudi.

**Franja 60–90 dies (opcional).** Allargaria la pregunta de l'antelació. Costa 30 dies més de separació: 4 dates menys de desenvolupament i 4 menys de test. Decisió 12: de moment no; només s'hi recorre si les tres franges no donen prou informació sobre l'antelació.

### 1.5 Partició i mostreig

```
2025                                                    2026
G  F  M  A  M  J  J  A  S  O  N  D  |  G  F  M  A  M  J
   [------ desenvolupament ------]      [---- test ----]
   5-2                         5-11     7-1         29-4
                                 <-63 d->            etiquetes fins al 28-6
```

| Bloc | Dates de referència | Nombre | Les etiquetes arriben fins a | Fallades aproximades |
| --- | --- | --- | --- | --- |
| Desenvolupament | 5-2-2025 … 5-11-2025 | 40 | 4-1-2026 | unes 4.000 **[S]** |
| Separació | 12-11-2025 … 31-12-2025 | 8, no es fan servir | — | — |
| Test | 7-1-2026 … 29-4-2026 | 17 | 28-6-2026 | unes 2.400 (Q1 **[C]** i Q2 de la web) |

- La primera data és el 5-2-2025 perquè calen 30 dies anteriors per als canvis. L'última és el 29-4-2026 perquè calen 60 dies posteriors.
- **Validació creuada:** `GroupKFold` per `serial_number` amb K = 3, només al desenvolupament. Serveix per triar hiperparàmetres, variables i models.
- **Atenció:** les etiquetes de les últimes dates de desenvolupament (finals d'octubre i novembre) cauen en les setmanes de Nadal, amb menys fallades. Com es tracten es decideix amb l'anàlisi de 1.5 bis.
- **Limitació de la validació creuada:** separa per disc però no per temps. Les fallades en lot (46 fallades d'un sol model Toshiba a l'octubre de 2025 i 2 al novembre **[C]**) queden repartides entre grups. Per això esperem que el test doni xifres pitjors que la validació creuada, i aquesta diferència és un resultat que es presenta en una frase.

**Avaluació extra «disc no vist».** Proposava fer-la només amb discs nous al test, però tenen massa poques fallades (21 de les 1.030 del Q1 2026 **[C]**). Alternativa equivalent i amb moltes més dades: **mostrejar els negatius per disc**. Si l'entrenament només veu un 15 % dels discs sense fallada, el 85 % dels discs del test no han aparegut mai a l'entrenament, i els discs que van ser positius al desenvolupament ja no hi són. El resultat estricte és el test restringit a aquests discs.

**Mostreig dels negatius i correcció:**

| Estrat | Què es conserva | Pes |
| --- | --- | --- |
| Files amb fallada dins de 60 dies | Totes (unes 33.000 **[S]**) | 1 |
| Resta de files | Discs triats per un hash del número de sèrie (15 %, uns 50.000 discs) i una de cada deu dates: π ≈ 1,5 %, unes 200.000 files **[S]** | 1/π |

- **Correcció de les probabilitats:** s'entrena sense pesos i es corregeixen les odds, p = qπ / (qπ + 1 − q), com ja fa `model.py` **[C]**. Val per a tots els models, inclosos k-NN i SVM. Per a la logística equival a sumar log π a la constant.
- **Submostres per cost:** k-NN amb 40.000 negatius i SVM amb 15.000, com ara.
- **Avaluació en validació creuada:** tots els positius del grup i una mostra ponderada de negatius (com ara).
- **Avaluació al test:** tota la població, data a data, perquè el pressupost d'alertes es defineix sobre el parc sencer.

**Memòria.** La mostra d'entrenament ocupa desenes de MB. El test sencer són 5,8 M de files; es puntua data a data (340.000 files cada vegada) i es desen només les prediccions. Res no exigeix més d'1–2 GB **[S]**.

### 1.5 bis Anàlisi preliminar de Nadal (decisió 3)

**Per què.** La troballa 1 diu que les fallades cauen les tres últimes setmanes de desembre. Abans de fixar la partició i `build.py` volem saber com incorporar aquest efecte sense que el test influeixi en la decisió, i tenir-ho documentat per a la diapositiva 2.

**Regla per no contaminar el test.** L'anàlisi només fa servir dades fins al 4-1-2026 (l'últim dia que fan servir les etiquetes de desenvolupament). Amb sis trimestres hi ha dues mostres de Nadal: el final de Nadal de 2024 (primeres setmanes del Q1 2025) i el Nadal de 2025 sencer. Els recomptes setmanals del Q1 2026 de l'annex A es van mirar abans de fixar la partició; es deixa escrit i no es fan servir per a cap decisió més.

**Preguntes:**

| | Pregunta | Com es mira | Dades |
| --- | --- | --- | --- |
| **N1** | Quina mida i forma té la caiguda? | Fallades per setmana i taxa per disc actiu, d'octubre de 2025 al 4-1-2026, i de gener–febrer de 2025 | `drives.parquet` |
| **N2** | Hi ha acumulació després de festes? | Les primeres setmanes de gener de 2025 tenen més fallades que la mitjana del trimestre? | Q1 2025 |
| **N3** | És retard en la retirada o menys fallades reals? | Per als discs que fallen al desembre, comparats amb els d'octubre: dies absents abans de la fallada i dies entre la primera pujada d'un comptador i la fallada. Si s'allarguen al desembre, és retard en la retirada | Panel Q4 2025 |
| **N4** | Quines etiquetes de desenvolupament toca? | Dates de referència la finestra de les quals cobreix les festes: a 60 dies, aproximadament les de finals d'octubre i novembre; a 30 dies, només les últimes. Quantes fallades «falten» respecte d'octubre | Capa 3 |
| **N5** | Afecta les variables del primer test? | Les dates de test de gener llegeixen 30 dies de desembre. Mirar si els dies d'absència (GO) i els canvis a 30 dies tenen una distribució diferent al desembre | Panel Q4 2025 |

**Opcions, que es trien amb el resultat:**

| Opció | Què és | Quan té sentit |
| --- | --- | --- |
| O1. Deixar-ho com està | Comprovació de sensibilitat traient les dates de desenvolupament afectades (com ja diu 1.5) | Si l'efecte és petit o N4 toca poques fallades |
| O2. Excloure o rebaixar les dates afectades | Les dates de desenvolupament amb finestra sobre les festes no entren a l'entrenament, o hi entren amb menys pes | Si N3 diu que és retard: aquelles etiquetes són incorrectes |
| O3. Començar el test més tard | Saltar les primeres dates de gener si N2 mostra acumulació al gener de 2025 | Només es decideix amb 2025, no mirant gener de 2026 |
| O4. Variable de calendari | Indicador «la finestra cobreix festes» | Corregeix l'entrenament, però al test és constant: no ajuda a ordenar. Probablement no |

**Sortida.** Una figura de fallades per setmana amb les festes marcades (per a la diapositiva 2), una frase de conclusió i l'opció triada, escrites a `context_processament.md`. Es fa al notebook `01_dades.py` aquesta setmana, abans de D1, i necessita el Q1 2025 consolidat.

### 1.6 On s'executa cada cosa

| Feina | On | Per què |
| --- | --- | --- |
| Descàrrega, consolidació i taula de dates (capes 0–3) | Homelab | Disc i temps. Només cal fer-ho en una màquina |
| Entrenament, cerca d'hiperparàmetres i puntuació del test (capes 4–5) | Homelab, amb CPU | Feines llargues que es poden deixar de nit |
| Notebooks i figures | Qualsevol portàtil | Només llegeixen `drives.parquet` i les capes 4 i 5: menys d'1 GB **[S]** |

- **Compartir resultats:** les capes 4 i 5 i `drives.parquet` es passen entre els dos membres per fora del git (rsync o carpeta compartida), perquè les dades no es versionen. El company no necessita el panel.
- **Regla de memòria:** es manté encara que el homelab tingui més RAM. El codi s'entrega i ha de poder executar-se en una màquina normal.
- **Especificacions del homelab:** no les conec (nuclis, RAM, disc). Els temps de la secció 3.1 estan pensats per a 4 nuclis.

**GPU: no es fa servir (decisió 17).** Tot surt de scikit-learn amb CPU, perquè els dos membres ho puguin reproduir. Els motius: el coll d'ampolla del processament és llegir i escriure fitxers; logística, arbre, Naive Bayes i gradient boosting per histogrames ja són ràpids amb CPU; l'única via (`cuml.accel` de RAPIDS) dona resultats no idèntics i potser no té paquets per a Python 3.14, cosa que obligaria a un entorn a part. k-NN i SVM es queden amb submostra.

---

## Part 2. Models candidats i combinacions

### 2.1 Referències sense aprenentatge

| Referència | Definició | Per què hi és |
| --- | --- | --- |
| Atzar | Taxa de fallada | Línia base de la PR-AUC |
| R1, regla de Backblaze | Algun dels cinc comptadors > 0 | Marca el 7,8 % dels discs i conté el 65 % de les fallades a 30 dies **[C]**. No permet triar un pressupost |
| **R2, regla de canvis** | Algun comptador ha pujat en 7 dies | Marca el 0,72 % i conté el 35 % **[C]**. És el rival real del model |
| R3, puntuació senzilla | Suma de `log1p` dels comptadors i dels seus canvis | Ordena tots els discs, i es pot avaluar a qualsevol pressupost |
| Taxa per model i edat | Taxa de fallada observada al desenvolupament per model de disc i tram d'edat | El que sabria una asseguradora. És la referència del grup G0 i dels discs nets |
| Kaplan-Meier | Com a l'estudi | Comprovar el calibratge mitjà de les corbes |

### 2.2 Models del curs i adaptacions

| Model | Adaptació al problema | Hiperparàmetres | Paper al treball |
| --- | --- | --- | --- |
| **Regressió logística** | `log1p`, estandardització, indicadors d'absent. L2 per als comptadors correlacionats; L1 per veure quines variables sobreviuen | Penalització i C | **Model principal.** Forma natural del model de franges; coeficients interpretables com a odds |
| **Random forest** | Cap transformació necessària. Fabricant com a variable | Variables per partició, mínim per fulla | **Model principal.** Millor ordenació a l'estudi |
| **Gradient boosting** | `HistGradientBoostingClassifier` amb log loss. Absents com a NaN, model de disc com a categòric natiu. Early stopping amb un conjunt de validació **separat per disc** | Taxa d'aprenentatge, profunditat, mínim per fulla; nombre d'arbres per early stopping | **Model principal.** Probablement el millor ordenant **[S]**; contrast directe amb el random forest |
| Arbre de decisió | Poc profund perquè es pugui llegir | Profunditat, mínim per fulla | **Figura explicativa:** l'arbre redescobreix la regla de Backblaze? |
| Naive Bayes | El gaussià falla amb comptadors gairebé sempre a 0. Variants: Bernoulli (> 0 o no) i categòric per trams (0, 1–9, 10–99, ≥ 100, absent) amb suavitzat de Laplace | Variant i α | Taula comparativa. Bona història: de pitjor model a competitiu canviant la versemblança |
| k-NN | Estandardització. Reduir a 8–12 variables triades dins de cada grup d'entrenament | k i pesos | Taula comparativa. Il·lustra la maledicció de la dimensionalitat |
| SVM | Nucli RBF amb 15.000 files. La funció de decisió ordena; per tenir probabilitats cal Platt | C i γ | Taula comparativa |

**Notes per model:**

- **Naive Bayes.** Els comptadors estan correlacionats (187, 197 i 198 pugen junts). La hipòtesi d'independència compta la mateixa evidència diverses vegades i dona probabilitats massa extremes. L'ordenació pot ser bona i el calibratge dolent: cal mirar-ho per separat.
- **k-NN.** Amb k = 50 només hi ha 51 valors possibles de probabilitat, i a la part alta de l'ordenació hi ha molts empats. Els pesos per distància ho alleugen.
- **SVM lineal contra logística (opcional).** Mateixa família d'hiperplans, pèrdua diferent (hinge contra log loss). És una comparació de teoria barata.
- **Gradient boosting.**
  - **Early stopping sense fuites.** La partició interna per defecte (`validation_fraction`) tria files a l'atzar, i files del mateix disc quedarien a entrenament i validació. Es fa servir `fit(X, y, X_val=…, y_val=…)` **[C]** amb un grup de discs reservat dins del grup d'entrenament de cada fold.
  - **Desequilibri.** Mateixa mostra i mateixa correcció d'odds que la resta. Com que minimitza la log loss, les probabilitats ja surten raonablement calibrades **[S]**; es comprova igual que amb els altres.
  - **Per què encaixa amb la pregunta.** El random forest redueix variància fent la mitjana d'arbres profunds; el boosting redueix biaix sumant arbres poc profunds. Si el boosting guanya sobretot a la franja 30–60, on el senyal és feble, diu que el problema allà és de biaix, no de soroll.
  - **Opcional, fora de les diapositives:** `interaction_cst` permet un model sense interaccions (suma de funcions d'una variable). Comparar-lo amb el complet diu si el senyal necessita combinar variables (per exemple, fabricant × comptador). `monotonic_cst` permet obligar el risc a créixer amb els comptadors d'error **[C]**; no és del temari.
- **Explicabilitat amb SHAP (tema Explainability, afegit el 23-11-2026).** TreeSHAP per al gradient boosting i el random forest, calculat sobre una mostra fixa (totes les alertes i uns milers de discs sense alerta). Comprovat: `shap` 0.54.1 funciona amb Python 3.14, scikit-learn 1.9.1 i `HistGradientBoostingClassifier` amb absents, i la suma de contribucions coincideix exactament amb la sortida en log-odds.
  - **Per grups.** Per la propietat d'eficiència, les contribucions sumen la predicció i es poden sumar per grup (G0, G1, GO, G2, G3). Per a cada alerta diu quina part del risc ve dels comptadors d'error de la regla de Backblaze. Complementa E1 (què aporta cada grup en conjunt) i E2 (en què es basa el model quan encerta una fallada neta).
  - **Gràfics:** beeswarm per franja (el pes dels comptadors baixa amb l'horitzó?) i waterfall d'unes quantes alertes reals per a l'aplicació (per què salta l'avís).
  - **Escala.** Al gradient boosting, SHAP explica els log-odds; la correcció per mostreig hi suma una constant i només mou el valor base. Al random forest explica la probabilitat abans de la correcció, que no és lineal: les contribucions es comparen entre variables, però no són punts de probabilitat real.
  - **Límits que cal dir.** Les variables correlacionades (nivell i canvi d'un comptador) es reparteixen el mèrit, i per això es resumeix per grups. SHAP explica el que ha après el model, no relacions causals. La logística no ho necessita (coeficients) i k-NN i SVM queden fora (KernelSHAP és lent i no aporta res a la història). LIME no es fa servir.
- **Random forest i error «out-of-bag».** L'error OOB es calcula per fila i les files d'un disc estan correlacionades, així que surt optimista. Comparar OOB, validació per disc i test temporal del mateix model és un experiment gratuït (2.4).

### 2.3 Combinacions

| Idea | Val la pena? | Justificació |
| --- | --- | --- |
| **Model de franges amb logística i amb random forest** | **Sí, imprescindible** | És la formulació principal |
| **Model sense comptadors contra model complet, ampliat a escala de 5 grups** | **Sí, imprescindible** | És l'experiment central |
| Dues etapes: regla i després model | **Sí, com a avaluació** | Avaluar per separat discs marcats i nets ja respon la pregunta. Entrenar dos models és opcional; un arbre ja pot fer aquesta partició tot sol |
| Mitjana de l'ordenació de logística i del millor conjunt d'arbres | Opcional, 1 hora | Fer la mitjana de models redueix la variància, i més si estan poc correlacionats |
| **Random forest contra gradient boosting** | **Sí** | Bagging (variància) contra boosting (biaix), amb les mateixes variables i franges. Surt d'E3 sense feina extra |
| Stacking amb logística a sobre | No | Guany incert, risc de fuites entre nivells i difícil d'explicar en 8 minuts |
| Model per fabricant | Opcional | Un sol experiment amb el model final |
| Una sola logística per a totes les franges (b2) | Opcional | Elegant, però no canvia la història |
| Horitzó com a variable (c) | No | Vegeu 1.4 |
| **Fora del temari:** regressió isotònica per calibrar | Opcional | Platt és una regressió logística d'una variable i sí que és del temari |
| **Fora del temari:** Kaplan-Meier per a l'esperança (Q2) | **Sí** | Ja es fa servir a l'estudi. S'explica com la proporció de discs que segueixen vius, corregida pels que surten de les dades |

### 2.4 Experiments

| | Experiment | Models | Prioritat |
| --- | --- | --- | --- |
| **E1** | Escala de variables (G0, +G1, +GO, +G2, +G3) × 3 franges | Logística i el millor conjunt d'arbres a D2 (random forest o gradient boosting) | Imprescindible |
| **E2** | Discs nets i marcats a la data de referència, i fallades netes i brutes | Models finals i referències | Imprescindible |
| **E3** | Comparació dels set models i les referències per franja, inclòs random forest contra gradient boosting | Tots | Imprescindible |
| **E4** | Franges (b) contra horitzons separats (a): monotonia, calibratge, ordenació | Logística, random forest, gradient boosting | Imprescindible |
| **E5** | Simulació d'alertes setmanals: corba de pressupost, antelació, cost | Model final, R2, R3 | Imprescindible |
| **E6** | Com canvia la xifra segons la validació: K-fold per fila, OOB, `GroupKFold` per disc, test temporal | Random forest | Recomanat; barat i molt defensable |
| **E7** | Esperança: supervivència observada per grup de risc predit, fins a 174 dies; esperança restringida a 60 dies per disc | Model final | Imprescindible (és una part de la pregunta) |
| E8 | Variants de Naive Bayes | — | Opcional |
| E9 | Model per fabricant, mitjana de models, gradient boosting sense interaccions, franja 60–90 (només si a l'antelació li falta informació) | — | Opcional |

### 2.5 Què arriba a la presentació

| A les diapositives | A la taula comparativa (una diapositiva) | Només exploració o diapositives de reserva |
| --- | --- | --- |
| Logística, random forest i gradient boosting per franges; referències R1, R2 i taxa per model i edat; arbre dibuixat | k-NN, SVM, Naive Bayes (millor variant), arbre | Variants de Naive Bayes, SVM lineal, model per fabricant, boosting sense interaccions, (b2) |

---

## Part 3. Hiperparàmetres i compromís realista

### 3.1 Procediment

1. Tot es decideix amb `GroupKFold` (K = 3) per disc dins del desenvolupament.
2. Tot el preprocessament (medianes, mitjanes, desviacions, llista de models freqüents, selecció de variables per a k-NN) s'ajusta dins de cada grup d'entrenament.
3. Es tria el valor amb millor mitjana de la mètrica de selecció. Si diversos valors queden dins d'una desviació estàndard del millor, es tria el model més simple.
4. Es reentrena amb tot el desenvolupament i es congela (29 de novembre).
5. **El test s'executa una sola vegada**, amb tots els models i experiments ja fixats. Després no es canvia res.

**Espais de cerca i recompte** (combinacions × K + 1, per franja):

| Model | Valors | Combinacions | × 3 + 1 | Franges | Entrenaments | Temps **[S]** |
| --- | --- | --- | --- | --- | --- | --- |
| Logística | penalització {L1, L2} × C {0,01; 0,1; 1; 10} | 8 | 25 | 3 | 75 | 30 min |
| k-NN | k {5, 25, 100, 400} × pesos {uniforme, 1/d} | 8 | 25 | 3 | 75 | 30 min |
| Naive Bayes | gaussià; Bernoulli i categòric × α {0,1; 1; 10} | 7 | 22 | 3 | 66 | 5 min |
| SVM RBF | C {0,1; 1; 10} × γ {0,003; 0,03; 0,3} | 9 | 28 | 1 (7–30) i 2 finals | 30 | 1 h |
| Arbre | profunditat {3, 5, 8, 12} × mínim per fulla {20, 100} | 8 | 25 | 3 | 75 | 5 min |
| Random forest | variables per partició {√d, d/3} × mínim per fulla {1, 5, 20} | 6 | 19 | 3 | 57 | 2–3 h |
| Gradient boosting | taxa d'aprenentatge {0,05; 0,1} × profunditat {3, 5} × mínim per fulla {20, 100}; fins a 1.000 arbres amb early stopping | 8 | 25 | 3 | 75 | 1 h **[S]** |
| **Total** | | | | | **453** | **unes 6 h** |

- **Per què aquests valors.** Cada hiperparàmetre es mou del costat rígid al flexible del compromís biaix-variància amb 3 o 4 punts en escala logarítmica. γ es centra en 1/d (≈ 0,03 amb unes 45 variables estandarditzades). El nombre d'arbres no s'ajusta (més arbres no sobreajusten): 100 durant la cerca i 300–500 al final.
- **Gradient boosting:** el nombre d'arbres és l'hiperparàmetre més important, i no es busca a la graella: el fixa l'early stopping amb discs reservats (5–15 iteracions sense millora). La taxa de 0,1 és la que recomanen les diapositives; 0,05 comprova si més arbres i passos més petits ajuden. La profunditat queda dins del rang habitual de 2 a 7. Es guarda el nombre d'arbres triat a cada fold i, en reentrenar amb tot el desenvolupament, es fa servir la mitjana.
- **SVM:** s'ajusta només a la franja 7–30 i es reutilitzen C i γ a les altres, per cost.
- **E1** afegeix uns 120 entrenaments (5 grups × 3 franges × 2 models × 4), unes 2 hores **[S]**. Si el conjunt d'arbres triat és el gradient boosting, és més ràpid.
- Els temps són estimacions. El primer dia de models cal cronometrar un entrenament de cada tipus i ajustar el pla.

### 3.2 Mètriques

L'operador revisa el parc un cop per setmana i pot actuar sobre un nombre limitat de discs.

| Mètrica | Definició | Ús |
| --- | --- | --- |
| **Detecció amb pressupost k** | A cada data, s'avisen els k % de discs amb més risc. Percentatge de files positives de la franja o horitzó que hi cauen | **Principal** per comparar models. k = 1 % per a la selecció |
| Precisió de les alertes | Dels discs avisats, quants fallen dins de l'horitzó | Sempre al costat de la detecció |
| Corba detecció–pressupost | Detecció per a k de 0,01 % a 10 %, eix logarítmic | Figura de l'aplicació |
| PR-AUC i ROC-AUC | Com ara | Secundàries; continuïtat amb l'estudi |
| Log loss | Després de la correcció per mostreig | Qualitat de les probabilitats |

**Per què la detecció amb l'1 % i no la PR-AUC per triar.** Als resultats actuals la PR-AUC del random forest a 30 dies val 0,079 ± 0,017 entre grups (variació del 21 %) i la detecció amb l'1 % val 0,507 ± 0,031 (6 %) **[C]**. La segona és més estable i és la que entén l'operador.

**Quin pressupost és realista.** L'1 % del parc són unes 3.400 alertes per setmana, quan en fallen unes 80 **[C]**. Com a càrrega de treball és molt. Proposta:

- 1 % per comparar models (estable, continuïtat amb l'estudi).
- 0,1 % (unes 340 alertes) com a punt «equip petit».
- La corba sencera, perquè cadascú hi llegeixi el seu punt.
- A la simulació, comptar **alertes noves per setmana**: un disc avisat continua a la llista les setmanes següents, i la càrrega real és molt més baixa que k.

**Simulació per disc (aplicació, E5).** Cada dimecres del test s'avisen els k % amb més risc a 30 dies. Per a cada disc que falla al període:

- **Detectat amb marge d** si ha estat avisat almenys d dies abans de fallar (d = 1, 7, 14).
- **Antelació:** dies entre el primer avís i la fallada.
- **Falsa alarma:** disc avisat que no falla en els 60 dies següents a l'últim avís.
- **Classe d'urgència** (Part 0): per a cada classe, quants discs fallen dins de la seva franja. És la resposta al «quan».

**Cost.** Amb C_FN el cost d'una fallada imprevista i C_FP el d'actuar sobre un disc, i ρ = C_FN / C_FP:

- Benefici d'avisar k discs = TP · (ρ − 1) − FP.
- **Avisar un disc surt a compte si la seva probabilitat de fallar supera 1/ρ.**
- Amb una precisió del 8 % cal ρ > 12,5; amb l'1,4 % (horitzó de 7 dies), ρ > 72.
- Figura: benefici segons el pressupost per a ρ = 5, 20 i 100.
- Lectura probable **[S]**: Backblaze reparteix les dades amb redundància i una fallada aïllada no perd dades, així que ρ és baixa per a «substituir» i el model serveix sobretot per a accions barates (copiar dades, programar una prova, prioritzar la revisió).

### 3.3 Calibratge

| Ús de la sortida | Cal calibrar? |
| --- | --- |
| Ordenar discs i avisar els k primers | No. L'ordre no canvia amb una transformació creixent |
| Corba de probabilitat acumulada (requisit del professor) | Sí |
| Llindar per cost (p > 1/ρ) | Sí |

- **Com:** correcció per mostreig primer. Després, Platt (logística d'una variable sobre el log-odds) ajustat amb les prediccions de validació creuada del desenvolupament, que el model no ha vist en entrenar. S'aplica tal qual al test.
- **Gradient boosting:** com que minimitza la log loss, potser ja surt calibrat després de la correcció per mostreig. Es mira el diagrama abans d'aplicar-hi Platt; si no millora la log loss en validació creuada, no es calibra.
- **Comprovació:** diagrama de fiabilitat en escala logarítmica i mitjana predita contra Kaplan-Meier, com a l'estudi (la logística sobreestimava un 25–38 % i el random forest infraestimava un 4–18 %).
- **Límit:** si la taxa de fallada canvia entre 2025 i 2026, el calibratge mitjà es desviarà al test. No es pot corregir sense mirar el test; es presenta com a resultat. La taxa anualitzada va de l'1,1 % al Q4 2025 i l'1,2 % al Q1 2026 **[C]** a l'1,73 % al Q2 2026 (web), així que és probable.

### 3.4 Estabilitat i incertesa

- **Entre grups:** mitjana i desviació de la validació creuada, com ara.
- **Entre dates:** la mètrica per a cada una de les 17 dates de test, en una sola figura.
- **Bootstrap per disc al test:** es remostregen discs amb reposició (1.000 rèpliques) i es recalculen les mètriques i les **diferències aparellades** entre models. Les alertes es deixen fixes, perquè són la decisió presa sobre el parc real. Interval del 95 % per percentils. Es fa al notebook amb la capa 5.
- **Una diferència només es presenta com a tal si l'interval no inclou el zero.**

---

## Part 4. Teoria i defensa

### 4.1 Decisions i conceptes del curs

Els números de diapositiva són la pàgina del PDF de `../teoria/`. Les fórmules no surten en extreure'n el text; les he contrastat amb `../formulari.md`.

| Decisió | Concepte | On |
| --- | --- | --- |
| Entrenar cada franja amb log loss | Sortida Bernoulli; màxima versemblança equival a minimitzar la log loss | Statistical Learning 8–11, 13; Linear Models 27 |
| Model de franges | La versemblança és un producte de Bernoullis (una per disc i franja en risc); el logaritme la converteix en suma de log loss | Statistical Learning 8–9 |
| Mètrica segons la decisió de l'operador | Una mètrica és «més propera al punt de vista del negoci» que la pèrdua | Statistical Learning 15 |
| No fer servir l'accuracy | Amb un 0,1 % de positius, dir sempre «no falla» encerta el 99,9 % | Statistical Learning 17; formulari §1 |
| Precisió i detecció amb pressupost | Precisió penalitza falses alarmes; recall, fallades no detectades. Fixar k és fixar el llindar | Statistical Learning 17–18 |
| Corba PR a més de la ROC | La PR no fa servir els TN i la seva base és la proporció de positius | Statistical Learning 19–20 |
| Calibrar no canvia l'ordenació | L'AUC només depèn de l'ordre; la log loss, no | formulari §1 |
| Desenvolupament, validació creuada i test únic | Train/val/test i K-fold; el test només estima el rendiment final | Statistical Learning 25–28 |
| Preprocessament dins de cada grup; test temporal | El test no pot influir en cap decisió; en dades temporals, entrenar amb el passat | formulari §1 |
| Validació agrupada per disc | Si el mateix disc és a entrenament i validació, es sobreestima el rendiment | Statistical Learning 25 |
| Diagnòstic entrenament contra validació | Sobreajust i subajust | Statistical Learning 24, 29–30 |
| Espais de cerca de rígid a flexible | Compromís biaix-variància; no s'observa, es mira l'error de validació | Statistical Learning 32–34; formulari, resum final |
| Correcció per mostreig dels negatius | Mostrejar negatius només canvia el prior; el posterior es corregeix amb Bayes | Naive Bayes 4, 6, 9 |
| Estandarditzar abans de regularitzar | La penalització depèn de l'escala | Linear Models 29, 31; formulari §2 |
| L2 per als comptadors d'error | Variables correlacionades donen coeficients molt grans sense Ridge | Linear Models 29 |
| L1 per triar variables | Lasso posa coeficients exactament a zero | Linear Models 31, 33 |
| Interaccions fabricant × SMART | Afegir productes de variables manté el model lineal en els paràmetres | Linear Models 23 |
| Coeficients com a odds | Pujar x en 1 multiplica les odds per e^β | formulari §2 |
| k del k-NN | k petita sobreajusta, k gran subajusta | Knn 8 |
| Estandarditzar i reduir variables al k-NN | La distància depèn de l'escala; maledicció de la dimensionalitat | Knn 7, 25 |
| Pesos per distància | Els veïns propers pesen més | Knn 18 |
| k-NN i SVM amb submostra | Predir amb k-NN és lent: distància a tots els punts | Knn 6, 25 |
| Naive Bayes gaussià falla | La versemblança gaussiana no s'assembla a un comptador gairebé sempre a 0 | Naive Bayes 12 |
| Naive Bayes categòric amb Laplace | Valors poc freqüents no han de donar versemblança zero | Naive Bayes 12, 17 |
| Naive Bayes mal calibrat | Hipòtesi d'independència amb SMART correlacionats: evidència comptada dues vegades | Naive Bayes 11; formulari §4 |
| C de l'SVM | Preu de cada violació del marge | SVM 12 |
| γ del nucli RBF | Abast de la similitud; γ gran és molt local | SVM 18, 22 |
| SVM amb RBF sense calcular variables noves | Truc del nucli i Mercer | SVM 20–24 |
| Profunditat i mínim per fulla | Criteris d'aturada triats per validació creuada | Decision Trees 15 |
| Arbre poc profund com a explicació | Fàcil d'interpretar, poc poder predictiu | Decision Trees 32 |
| Random forest | Bagging redueix la variància; triar variables a l'atzar baixa la correlació entre arbres | Decision Trees 24, 26 |
| No ajustar el nombre d'arbres | No sobreajusta amb molts arbres | Decision Trees 33 |
| OOB optimista amb files correlacionades | L'OOB prediu les files fora de la bossa, però hi ha altres files del mateix disc a dins | Decision Trees 27 |
| Importància per permutació i no per impuresa | La d'impuresa afavoreix variables amb molts valors diferents | Decision Trees 29–30 |
| Mitjana de logística i random forest | Conjunt de models de tipus diferent; fer la mitjana redueix la variància | Decision Trees 21; Statistical Learning 30 |
| Afegir gradient boosting | Molts classificadors febles en sèrie donen un classificador fort | Gradient Boosting 4 |
| Random forest contra gradient boosting | Bagging redueix la variància; boosting segueix el descens del gradient i redueix el biaix pas a pas | Gradient Boosting 6–7, 10; Decision Trees 24 |
| Log loss al boosting | En classificació, cada arbre aprèn els pseudo-residus de la log loss sobre els log-odds | Gradient Boosting 16 |
| Profunditat del boosting entre 3 i 5 | Arbres poc profunds, habitualment de profunditat 2 a 7 | Gradient Boosting 9 |
| Taxa d'aprenentatge 0,05 i 0,1 | Massa gran és inestable, massa petita no convergeix; 0,1 funciona bé amb molts arbres | Gradient Boosting 11 |
| Nombre d'arbres per early stopping amb discs reservats | És l'hiperparàmetre més important; es pot triar per validació creuada o early stopping | Gradient Boosting 12 |
| Versió per histogrames de scikit-learn | Implementacions diferents: particions per histogrames, com LightGBM | Gradient Boosting 19 |
| Entrenament més lent que el random forest | Els arbres del boosting s'entrenen en sèrie; els del random forest, en paral·lel | Gradient Boosting 21; Decision Trees 33 |
| Bootstrap per disc | Mostreig amb reposició, aplicat a discs i no a files | Decision Trees 23 |
| SHAP per explicar el gradient boosting i el random forest | Valors de Shapley: contribució marginal mitjana de cada variable; TreeSHAP per a models d'arbres | Explainability 12–17 |
| Sumar SHAP per grups de variables | Eficiència: els valors sumen la predicció menys el valor base | Explainability 14, 16 |
| SHAP en log-odds al boosting i en probabilitat al random forest | Els valors són additius en l'espai de sortida del model | Explainability 18 |
| Beeswarm i waterfall | Explicació global a partir de les locals; descomposició d'una predicció | Explainability 6, 19–21 |
| Resumir per grups i no per variable | Amb variables correlacionades és difícil repartir la importància; l'explicació no és causal | Explainability 23 |

### 4.2 Preguntes probables i resposta breu

| Pregunta | Resposta |
| --- | --- |
| Per què no l'accuracy? | Amb un 0,1 % de fallades, dir sempre «no falla» encerta el 99,9 % i no detecta res. Mirem detecció i precisió amb un pressupost |
| El model no aprèn simplement la regla de Backblaze? | És l'angle propi del treball (Q0). Resposta amb E1 i E2: quant queda sense els comptadors, en discs nets i en fallades netes |
| Per què una precisió del 8 % és útil? | La taxa base és 0,17 %: és 45 vegades millor que triar a l'atzar. Si actuar surt a compte depèn de la ràtio de costos: cal que ρ > 1/precisió |
| Per què no barrejar files del mateix disc? | Són gairebé idèntiques d'una setmana a l'altra. El model reconeixeria el disc en lloc de generalitzar. Ho mostrem a E6 |
| Per què una separació de 60 dies? | L'etiqueta d'una data mira 60 dies endavant. Sense separació, l'entrenament veuria fallades del període de test |
| Per què el test dona pitjor que la validació creuada? | La validació separa discs però no temps: comparteix lots i època. El test és el cas real |
| Per què franges i no un model per horitzó? | Corbes que no baixen mai, i saber què es pot predir entre 30 i 60 dies sense barrejar-ho amb les fallades imminents |
| Si mostregeu els negatius, les probabilitats no surten inflades? | Sí, i es corregeixen: mostrejar només canvia el prior |
| Què feu amb els SMART que falten? | Falten per fabricant, no a l'atzar. Indicador d'absent i imputació amb valors del desenvolupament |
| Per què Naive Bayes va tan malament? | Comptadors gairebé sempre a 0 no són gaussians, i estan correlacionats |
| Per què no una xarxa neuronal? | No és del temari, i l'objectiu és comparar models i respondre la pregunta, no el millor rendiment. El gradient boosting ja fa de model fort |
| Per què el gradient boosting guanya (o no) al random forest? | Boosting redueix biaix i bagging, variància. On guanya el boosting, el problema era de biaix. Si empaten, el límit és el senyal que hi ha a les dades, no el model |
| L'early stopping no fa trampa amb el test? | No: s'atura amb un grup de discs reservat dins de l'entrenament de cada fold. El test no hi intervé, ni tampoc files del mateix disc |
| Això no ho ha fet ja Backblaze o algú altre? | Backblaze publica estadístiques i quins SMART mira. Hi ha articles externs de predicció. El nostre angle és quant del senyal és la definició de l'etiqueta, per franges i amb pressupost, amb dades de 2025–2026 **[S]** (cal repassar la bibliografia un cop) |
| Per què cauen les fallades al desembre? | L'etiqueta es posa quan es retira el disc, i per festes es retiren menys discs. Reforça que l'etiqueta és una decisió operativa |
| Els discs que desapareixen de les dades, què són? | La majoria tornen (85 %). Tenen 18 vegades més risc de fallar; per això els dies d'absència són una variable |
| Com sabeu que una diferència entre models és real? | Bootstrap per disc al test, amb diferències aparellades |
| Què vol dir «esperança» si només mireu 60 dies? | Per disc, només podem donar els dies de vida esperats dins dels 60 dies. Per això ensenyem la supervivència observada per grup de risc fins a 174 dies, que no extrapola res |
| Què vol dir «quan»? | La classe d'urgència: en quina franja el risc del disc supera el llindar |
| El resultat de k-NN i SVM depèn de la submostra? | S'entrenen amb submostra per cost; no són models principals. La variació entre grups de la validació creuada dona una idea de l'estabilitat |
| Què faríeu amb més temps? | Franja 60–90 dies, dos anys de dades per a la deriva, i el motiu de retirada si Backblaze el publiqués |

---

## Part 5. Pla de treball i presentació

### 5.1 Fases i punts de decisió

El repartiment encara no està decidit. **A** (dades, validació i aplicació) i **B** (models i experiment central) són dos paquets de feina equilibrats, no persones; s'assignen quan l'abast estigui acordat. Cadascú revisa la feina de l'altre abans de cada fita.

| Fase | Dates | A | B | Fita |
| --- | --- | --- | --- | --- |
| **0** | dj. 12 – dg. 15 nov. | Baixar i consolidar Q1–Q3 2025, Q1–Q2 2026. Arreglar `consolidate.py` (columnes fixes, taula de discs per trimestre) | Regenerar `landmarks.parquet` amb el `build.py` actual i comprovar 323 / 1.422 / 2.331. Cronometrar un entrenament de cada model | Reunió: acordar aquest pla i les decisions obertes |
| **1** | dl. 16 – dg. 22 nov. | Anàlisi preliminar de Nadal (1.5 bis) i fixar la partició. `build.py` nou: dates setmanals, variables per grup, presència, partició, mostra d'entrenament. Notebook de dades | Marc d'avaluació: mètriques amb pressupost, referències R1–R3 i taxa per model i edat, bootstrap. Provat amb el Q4 | **D1** |
| **2** | dl. 23 – dg. 29 nov. | Model de franges i comparació amb horitzons separats (E4). Calibratge. E6 | Cerca d'hiperparàmetres dels set models (E3), amb el gradient boosting primer. Variants de Naive Bayes | **D2:** models congelats; es tria el conjunt d'arbres per a E1 |
| **3** | dl. 30 nov. – dg. 6 des. | Simulació d'alertes, classes d'urgència, antelació i cost (E5). Esperança per grup de risc (E7). Execució del test | E1 i E2 en validació creuada; arbre dibuixat; importància per permutació | **D3:** test executat (dv. 4), figures tancades |
| **4** | dl. 7 – ds. 12 des. | Diapositives 1–3 i 7. Neteja del codi i README | Diapositives 4–6. Diapositives de reserva per a preguntes | Assajos dc. 9 i dv. 11. Entrega ds. 12 |
| Marge | dg. 13 des. | | | Límit 23:59 |

**Punts de decisió:**

| | Quan | Condició | Si no es compleix |
| --- | --- | --- | --- |
| **D1** | dg. 22 nov. | La taula nova existeix, els recomptes del Q4 quadren amb l'estudi i les referències donen xifres coherents | Passar a quatre trimestres (Q3 2025 – Q2 2026) i variables de canvi només a 7 dies. Si tampoc: mantenir el Q4 sol amb el disseny de l'estudi i centrar-se en E1 i E2 amb validació creuada |
| **D2** | dg. 29 nov. | Els set models tenen hiperparàmetres triats i el model de franges funciona | k-NN i SVM es queden amb una configuració fixa (el gradient boosting té prioritat sobre ells). Si les franges ordenen clarament pitjor que els horitzons separats, es presenta (a) per ordenar i (b) per a la corba |
| **D3** | dv. 4 des. | Test executat una vegada | Res d'opcional a partir d'aquí. Si E1 no dona un gradient clar, la conclusió és «el senyal és la regla» i es reforcen E5 i E6 |

A partir del 29 de novembre no s'afegeixen variables ni models. A partir del 4 de desembre no es torna a executar el test.

### 5.2 Repartiment per a la presentació

Proposta provisional, lligada als paquets A i B:

- **A presenta les diapositives 1–3 i 7** (pregunta, dades i etiqueta, disseny, quins i quan): 3 min 50 s.
- **B presenta les diapositives 4–6** (probabilitat, de què ens podem refiar, esperança): 3 min 40 s.
- **La diapositiva 8** (conclusions, 30 s) es reparteix.
- Cadascú ha de poder respondre les preguntes de la part de l'altre: a l'assaig del dia 11, intercanvieu-vos les preguntes de 4.2.

### 5.3 Esquema de 8 minuts

L'ordre segueix les parts de la pregunta enviada.

| # | Temps | Qui | Missatge | Figura clau |
| --- | --- | --- | --- | --- |
| 1 | 0:40 | A | La pregunta en tres parts: probabilitat, esperança, quins substituir i quan. I un avís: qui decideix que un disc ha fallat mira els mateixos SMART que el model | Títol, la pregunta, i el parc en tres xifres |
| 2 | 1:00 | A | L'etiqueta és una decisió operativa: una regla senzilla ja troba moltes fallades, i les fallades baixen per Nadal | Fallades per setmana (octubre 2025 – març 2026) i cobertura de R1 i R2 |
| 3 | 1:00 | A | Com evitem enganyar-nos: test en el temps, separació de 60 dies, validació per disc, franges | Línia de temps de la partició i esquema d'una data amb les tres franges |
| 4 | 1:10 | B | **Probabilitat.** Set models del curs, tres franges; random forest contra gradient boosting; la corba acumulada és monòtona i calibrada | Barres de detecció amb l'1 % per model i franja, amb intervals; corbes acumulades de tres discs |
| 5 | 1:30 | B | **De què ens podem refiar.** D'on ve el senyal i fins a quina antelació | Escala de variables × franja (E1), i al costat discs nets contra marcats (E2) |
| 6 | 1:00 | B | **Esperança.** Què els passa als discs segons el risc que els donem | Supervivència observada per grup de risc al test, fins a 174 dies (E7) |
| 7 | 1:10 | A | **Quins i quan.** Pressupost d'alertes, classes d'urgència i marge real | Corba detecció–pressupost contra R2 i R3; histograma de l'antelació |
| 8 | 0:30 | A i B | Resposta a cada part de la pregunta en una frase, i els límits | Taula de quatre files: probabilitat, esperança, quins i quan, què no podem afirmar |

**Diapositives de reserva** per a preguntes: arbre dibuixat, E6 (validació), calibratge, cost segons ρ, Naive Bayes, importància per permutació, validació creuada contra test.

### 5.4 Notebooks de marimo

| Notebook | Propòsit | Llegeix |
| --- | --- | --- |
| `01_dades.py` | Parc, fallades per setmana, absències, SMART per fabricant, regles R1 i R2 | `drives.parquet`, mostra de dates |
| `02_validacio.py` | Partició, recomptes per franja, E6 | Capes 3–5 |
| `03_models.py` | Corbes de validació per hiperparàmetre (inclosa la corba d'early stopping del boosting), taula dels set models, E4 | Resultats de la cerca, capa 5 |
| `04_senyal.py` | E1, E2, arbre dibuixat, importància per permutació, SHAP per grups (beeswarm per franja i waterfall d'alertes) | Capa 5 |
| `05_aplicacio.py` | Corba de pressupost, classes d'urgència, antelació, cost amb controls per a k i ρ, calibratge, corbes de risc, supervivència per grup de risc | Capa 5, `drives.parquet` |

Els scripts pesats (consolidar, construir la taula, entrenar i puntuar) continuen a `dataset/backblaze/scripts/`. `05_aplicacio.py` es pot obrir com a aplicació durant les preguntes.

### 5.5 Riscos

| Risc | Probabilitat | Mitigació |
| --- | --- | --- |
| Unir trimestres dona problemes (esquema, memòria) | Mitjana | Fer-ho la setmana 0. Els dos canvis de `consolidate.py` ja estan identificats. Pla B a D1 |
| Abast massa gran | Alta | Llista d'imprescindibles (E1–E5 i E7). Tot l'opcional es talla a D2 i D3 |
| «Esperança» s'entén com a vida total del disc | Mitjana | Dir a la diapositiva 6 que és supervivència observada a 174 dies, no una extrapolació |
| Mirar el test més d'una vegada | Mitjana | Execució única el 4 de desembre, anotada a `context_processament.md` |
| E1 sense resultat clar | Mitjana | Qualsevol resultat respon la pregunta (taula de la Part 0) |
| Temps de còmput | Baixa–mitjana | Cronometrar el dia 15; repartir entre dues màquines; SVM i k-NN amb submostra |
| Efecte de Nadal a les etiquetes de desenvolupament | Baixa | Anàlisi preliminar (1.5 bis) abans de D1, només amb dades fins al 4-1-2026 |
| Fallades en lot dominen una mètrica | Mitjana | Mètrica per data i bootstrap per disc; esmentar-ho |
| La taxa de fallada canvia entre 2025 i 2026 | Alta | No afecta l'ordenació; el calibratge es presenta com a resultat |
| GO resulta ser una altra fuita | Mitjana | Per això és un grup separat a l'escala |
| Early stopping del boosting amb la partició interna per defecte (fuita entre files del mateix disc) | Mitjana | Passar sempre `X_val` i `y_val` amb discs reservats; `early_stopping` mai amb `validation_fraction` |
| Les dues màquines donen resultats diferents | Baixa | `uv.lock`, llavors fixes, mostreig per hash del número de sèrie |
| Un membre no pot treballar una setmana | Mitjana | Cada fita deixa les capes 3–5 escrites; l'altre pot seguir amb els notebooks |

---

## Imprescindible i opcional

| Imprescindible | Opcional si sobra temps |
| --- | --- |
| Sis trimestres (mínim quatre) i partició temporal | Franja 60–90 dies |
| Taula setmanal amb grups G0–G3 i GO | Model per fabricant |
| Model de franges amb logística i random forest | Logística única per a totes les franges (b2) |
| Referències R1, R2, R3 i taxa per model i edat | Gradient boosting sense interaccions |
| Els set models amb cerca petita, inclòs el gradient boosting | Variants de Naive Bayes, SVM lineal |
| E1, E2, E3, E4, E5, E7 | E6 (recomanat), mitjana de models |
| Bootstrap per disc al test | Regressió isotònica |
| Calibratge amb Platt | Dades de 2024 |
| Classes d'urgència | |
| Anàlisi preliminar de Nadal (1.5 bis) | |

---

## Decisions obertes per a la parella

Estat a 24-11-2026 (reunió de la parella del 19-11 i decisions del 24-11). **Decidit** vol dir que no es torna a discutir; **provisional**, que es treballa amb la recomanació però falta confirmar-la; **pendent**, que encara no s'ha decidit.

| # | Decisió | Estat | Què s'ha acordat o què falta | Alternativa descartada o oberta |
| --- | --- | --- | --- | --- |
| 1 | Pregunta | **Decidit** | Tancada i acordada amb el professor. Q0 es manté com a angle propi; s'hi lliga el seguiment de la decisió 14 | — |
| 2 | Trimestres | **Decidit** | Sis: Q1 2025 – Q2 2026. La partició es deriva d'aquí (1.5) | Quatre trimestres, només com a pla B a D1 |
| 3 | Període de test i Nadal | **Decidit** (24-11) | Test del 7-1 al 29-4-2026. Etiquetes de desenvolupament tallades el 16-11-2025 (L + t ≤ 16-11-2025; franges posteriors censurades), sense variable de calendari. L'entrenament no fa servir cap informació posterior al tall, ni tan sols per saber que un disc segueix viu | Novembre–desembre de 2025 com a test: descartat |
| 4 | Formulació | **Decidit** | Franges (b1), amb horitzons separats (a) com a comparació | Només (a) |
| 5 | Freqüència de dates | **Decidit** | Setmanal (dimecres), aprimant negatius | Cada 2 setmanes |
| 6 | Variables d'absència (GO) | **Decidit** | Grup separat a l'escala de variables | No incloure-les |
| 7 | Mètrica de selecció | **Decidit** (24-11) | Detecció amb l'1 % d'alertes per seleccionar; PR-AUC i ROC-AUC com a secundàries | PR-AUC com a principal |
| 8 | Pressupost per a l'aplicació | **Pendent** | Proposta: corba sencera, amb 0,1 % i 1 % marcats | Només 1 % |
| 9 | Mostreig de negatius | **Pendent** | Proposta: per disc (15 %) i una de cada deu dates | Per fila |
| 10 | Models a les diapositives | **Decidit** | Logística, random forest i gradient boosting; la resta en una taula | Tots set amb el mateix pes |
| 11 | Model fora del temari | **Decidit** | Cap, tret de Kaplan-Meier per a l'esperança. SHAP és del temari (tema Explainability, afegit el 23-11-2026) i es fa servir per al gradient boosting i el random forest | — |
| 12 | Franja 60–90 dies | **Decidit** | De moment no. Queda com a recurs si a la pregunta d'antelació li falta informació | Sí des del principi |
| 13 | Repartiment dels paquets A i B | **Pendent** | Qui tingui el homelab, millor A | — |
| 14 | Seguiment amb el professor | **Pendent** | Proposta: ensenyar-li Q0 i la definició d'«esperança» a la propera classe | — |
| 15 | Definició d'«esperança» | **Pendent** | Proposta: supervivència observada per grup de risc, i esperança restringida a 60 dies per disc | Extrapolació amb risc constant; taula de vida per edat |
| 16 | Definició de «quan» | **Decidit** | Classes d'urgència per franja | Només el pressupost setmanal |
| 17 | Ús de la GPU | **Decidit** | No es fa servir. Tot amb scikit-learn i CPU | Experiment de k-NN i SVM amb tota la mostra: descartat |
| 18 | Paper del gradient boosting | **Decidit** | Model principal, al costat del random forest; E1 amb el millor dels dos a D2 | Només a la taula comparativa |
| 19 | Població a cada data | **Decidit** (24-11) | Discs vistos dins de [L−7, L] amb els últims valors (`days_since_seen`). Amb «fila el dia L» un 15 % de les fallades no sortien a la franja 0–7 | Només discs amb fila el dia L |

---

## Annex A. Comprovacions i suposicions

### Comprovat avui (12-11-2026)

| Què | Com | Resultat |
| --- | --- | --- |
| Trimestres publicats | Pàgina de dades de Backblaze | Fins al Q2 2026; mides a 1.1 |
| Condicions d'ús | Mateixa pàgina | Citar la font; no vendre les dades |
| Canvis d'esquema | `Drive_Stats_Schema_2018_Onward.csv` de Backblaze | Arriba fins al Q1 2024. `vault_id`, `pod_id`, `is_legacy_format` des del Q2 2023; `datacenter`, `cluster_id`, `pod_slot_num` des del Q3 2023 |
| Q1 2026 | Zip baixat a un directori temporal fora del repositori, llegit sense extreure i esborrat | 90 fitxers, 30.597.484 files, 1.030 fallades, una sola capçalera, mateixes 93 columnes SMART `_raw` que el Q4 2025 |
| Fallades per setmana, Q4 2025 | `drives.parquet` | 59 (5 dies), 107, 79, 131, 111, 88, 84, 56, 78, 58, 45, 33, 10, 6 (3 dies) |
| Fallades per setmana, Q1 2026 | Zip | 12 (4 dies), 66, 64, 71, 102, 76, 116, 53, 72, 82, 134, 73, 74, 35 (2 dies) |
| Discs que surten al Q4 | `drives.parquet` i zip | 1.904; 1.627 tornen al Q1 2026; 87 d'aquests hi fallen |
| Discs que fallen i tornen | Ídem | 1 de 945 |
| Discs nous al Q1 2026 | Ídem | 10.653; 21 de les 1.030 fallades |
| Dies absents al Q4 | `drives.parquet` | 10.641 discs (3,1 %); 254 dels 945 que fallen (27 %) |
| Regles a la data 8-10-2025 | `landmarks.parquet` | R1: 25.913 de 331.837 discs, 244 de 377 fallades a 30 dies. R2: 2.389 discs, 133 fallades |
| Comptadors abans de fallar | Panel Q4, discs que fallen | Algun > 0: 73 % l'últim dia, 67 % a 7 dies, 63 % a 30 (n = 467), 55 % a 60 (n = 156, només fallades de desembre). Algun puja l'última setmana: 53 % |
| Fabricants | `drives.parquet` | Seagate 115.668 discs i 408 fallades; Toshiba 113.105 i 247; WDC 83.897 i 95; HGST 27.247 i 194. 20 models amb 10 fallades o més sumen 899 de 945 |
| SMART 187 absent | `landmarks.parquet` | Seagate 1,7 %; resta ≈ 100 % |
| Estabilitat de mètriques | `results_summary.csv` | Random forest a 30 dies: PR-AUC 0,079 ± 0,017; detecció amb l'1 % 0,507 ± 0,031 |
| Scripts | Lectura de `consolidate.py`, `build.py`, `explore.py`, `model.py` | Punts de 1.1; correcció per mostreig ja implementada |
| Teoria | `../formulari.md` sencer i text de les set presentacions de `../teoria/` (Gradient Boosting afegida el 18-11) | Taula 4.1. Les fórmules de les diapositives són imatges i no s'han pogut llegir |
| Velocitat de descàrrega | Descàrrega del Q1 2026 | 1,3 GB en 78 s |
| API del gradient boosting | `HistGradientBoostingClassifier` a l'entorn del projecte (scikit-learn 1.9.1) | `fit` accepta `X_val`, `y_val`, `sample_weight_val`; té `categorical_features`, `interaction_cst`, `monotonic_cst`, `early_stopping`, `class_weight` |
| Acceleració amb GPU | Documentació de RAPIDS (`cuml.accel`) | Accelera k-NN, SVC, random forest i logística de scikit-learn sense canviar el codi; resultats no idèntics; provat amb scikit-learn 1.6–1.9.1; GPU amb capacitat de còmput 7.0 o superior; Linux |

No s'han llegit `../seguiment.md` ni `../examens/`.

### Suposicions que cal confirmar

| Suposició | Com confirmar-la | Quan |
| --- | --- | --- |
| Unes 4.000 fallades al desenvolupament | Consolidar Q1–Q3 2025 | Setmana 0 |
| Q1–Q3 2025 i Q2 2026 tenen les mateixes columnes | Recompte de columnes en consolidar | Setmana 0 |
| La taula de discs no cap en memòria amb sis trimestres | Mirar la memòria amb dos trimestres abans d'afegir-ne més | Setmana 0 |
| `pyarrow.dataset` falla o perd columnes amb esquemes diferents | Provar amb dos trimestres | Setmana 0 |
| Mides de les capes 3 i 4 i temps de còmput | Cronometrar | Setmanes 0–1 |
| L'efecte de Nadal és operatiu (menys retirades) | No es pot confirmar amb les dades; mirar si passa el desembre de 2024 en una frase del blog de Backblaze | Opcional |
| Els dies d'absència són un senyal i no una conseqüència de la sospita | E1 amb GO separat; mirar si l'absència precedeix la pujada dels comptadors | Setmana 3 |
| La ràtio de costos de Backblaze és baixa | Buscar com reparteixen les dades entre discs | Setmana 4, per a les preguntes |
| Ningú ha publicat aquesta anàlisi | Una cerca bibliogràfica breu | Setmana 1 |
| El professor accepta Q0 com a angle i la definició d'«esperança» | Preguntar-li | Propera classe |
| Backblaze ha publicat anàlisis de vida dels discs per edat | Cerca al seu blog | Setmana 1 |
| Especificacions del homelab | Mirar-les | Setmana 0 |
