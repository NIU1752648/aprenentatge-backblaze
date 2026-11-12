# Estudi de viabilitat: Backblaze Drive Stats (fallades de discs durs)

**Veredicte: sí que encaixa, amb una reserva important. El senyal és fort i té una aplicació clara (manteniment predictiu), però la major part prové de cinc comptadors SMART que la mateixa Backblaze fa servir per decidir que un disc ha fallat. El model pot acabar aprenent la regla de l'empresa en lloc de predir fallades reals, i això s'ha de discutir a l'informe.**

## 1. Resum del dataset

- **Font:** [Backblaze Drive Stats](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data). Lliure, amb la condició de citar Backblaze i no vendre les dades.
- **Trimestre fet servir:** quart trimestre de 2025, de l'1 d'octubre al 31 de desembre. Són 92 fitxers diaris, 1,16 GB comprimits i 12,2 GB descomprimits.
- **Contingut:** una fila per disc i dia, amb número de sèrie, model, capacitat, indicador de fallada (`failure` = 1 l'últim dia abans de fallar) i els valors SMART.
- **Mida:** 341.664 discs de 78 models, amb 945 fallades al trimestre.
- **Valors que falten:** alguns SMART només els reporten certs fabricants. Per exemple, `smart_187` i `smart_188` falten en el 66 % de les files, sempre en els mateixos models. Per això s'han fet servir indicadors de valor absent.

## 2. Construcció de l'objectiu

- **Dates de referència (`L`):** 8, 15, 22 i 29 d'octubre. Per a cada `L` es prenen els discs actius aquell dia, amb els seus SMART i el canvi dels comptadors d'error en els 7 dies anteriors.
- **Objectiu:** y_t = 1 si el disc falla en els `t` dies següents a `L`, per a t = 7, 30 i 60 dies.
- **Seguiment incomplet:** si un disc surt de les dades abans de `L + t` sense fallar, la seva etiqueta és desconeguda i s'exclou d'aquell horitzó.
- **Mida de la taula:** 1.337.354 files, de 335.996 discs diferents.

| Horitzó | Fallades | Etiqueta desconeguda | Files utilitzables | Taxa de fallada |
| --- | --- | --- | --- | --- |
| 7 dies | 323 | 50 | 1.337.304 | 0,024 % |
| 30 dies | 1.422 | 449 | 1.336.905 | 0,106 % |
| 60 dies | 2.331 | 3.092 | 1.334.262 | 0,175 % |

Les fallades es compten per data de referència, així que un disc que falla al novembre compta com a positiu a diverses dates. El seguiment incomplet és molt petit, perquè pocs discs surten de les dades sense fallar.

## 3. Resultats

**Mètode:**
- **Validació:** creuada amb 3 grups separats per disc (`GroupKFold`), perquè el mateix disc no aparegui a l'entrenament i a la prova.
- **Entrenament:** totes les fallades i una mostra aleatòria de 40.000 discs sense fallada (15.000 per a l'SVM).
- **Avaluació:** totes les fallades de prova i una mostra de 100.000 discs sense fallada, ponderada perquè les mètriques reflecteixin la proporció real.
- **Hiperparàmetres:** fixos, sense ajustar.

**Mètriques:**
- **ROC-AUC i PR-AUC.** La PR-AUC d'un model a l'atzar és igual a la taxa de fallada.
- **Pressupost de l'1 %:** el percentatge de fallades detectades marcant només l'1 % de discs amb més risc, és a dir, unes 3.350 alertes per data.

| Model | ROC-AUC 7 / 30 / 60 dies | PR-AUC 7 / 30 / 60 dies | Fallades detectades amb l'1 % d'alertes (7 / 30 / 60 dies) |
| --- | --- | --- | --- |
| A l'atzar | 0,50 | 0,0002 / 0,0011 / 0,0017 | 1 % |
| Regla de Backblaze (algun comptador d'error > 0) | 0,78 / 0,78 / 0,77 | 0,001 / 0,006 / 0,009 | 7 / 8 / 8 % |
| Regressió logística | 0,87 / 0,86 / 0,86 | 0,066 / 0,082 / 0,085 | 45 / 41 / 37 % |
| k-NN (k = 50) | 0,85 / 0,85 / 0,84 | 0,049 / 0,076 / 0,087 | 48 / 45 / 41 % |
| Naive Bayes gaussià | 0,75 / 0,73 / 0,71 | 0,003 / 0,012 / 0,016 | 16 / 3 / 14 % |
| Arbre de decisió (profunditat 6) | 0,76 / 0,80 / 0,78 | 0,016 / 0,044 / 0,058 | 45 / 42 / 40 % |
| **Random forest** | **0,92 / 0,89 / 0,88** | **0,061 / 0,079 / 0,101** | **57 / 51 / 46 %** |
| SVM RBF | 0,80 / 0,84 / 0,82 | 0,017 / 0,038 / 0,052 | 50 / 45 / 41 % |

- **Random forest:** és el millor. Amb l'1 % d'alertes, detecta el 46-57 % de les fallades. A 60 dies, la precisió de les alertes és del 8 %: 1 de cada 12 discs marcats falla.
- **Regla de Backblaze:** per a l'horitzó de 30 dies detecta el 64 % de les fallades, però marca el 7,8 % de tots els discs (unes 26.000 alertes per data), amb una precisió de només el 0,9 %.
- **Naive Bayes:** és molt inestable entre grups (desviació de la ROC-AUC de 0,2), perquè els comptadors SMART, quasi sempre a 0, no s'assemblen gens a una distribució normal.

**Sense els cinc comptadors d'error** (SMART 5, 187, 188, 197, 198 i els seus canvis en 7 dies):

| Model | ROC-AUC 7 / 30 / 60 dies | PR-AUC 7 / 30 / 60 dies |
| --- | --- | --- |
| Regressió logística | 0,71 / 0,70 / 0,71 | 0,001 / 0,008 / 0,011 |
| Random forest | 0,78 / 0,74 / 0,75 | 0,006 / 0,008 / 0,012 |

Sense aquests comptadors, la PR-AUC cau unes 10 vegades. El que queda prové sobretot del model i l'edat del disc.

**Importància per permutació** (random forest, 30 dies): les variables més importants són `smart_1` (taxa d'errors de lectura), `smart_5`, `smart_192`, `smart_240`, `smart_193` i el model del disc. `smart_1` té un significat diferent segons el fabricant, i cal tenir-ho en compte abans d'interpretar-la.

## 4. Probabilitat acumulada i Kaplan-Meier

Mitjana per data de referència. Les probabilitats estan corregides per la mostra de discs sense fallada amb què s'ha entrenat.

| Horitzó | Kaplan-Meier | Proporció simple | Regressió logística | Random forest |
| --- | --- | --- | --- | --- |
| 7 dies | 0,00024 | 0,00024 | 0,00033 | 0,00023 |
| 30 dies | 0,00106 | 0,00106 | 0,00132 | 0,00090 |
| 60 dies | 0,00174 | 0,00175 | 0,00227 | 0,00143 |

- **Kaplan-Meier i proporció simple:** coincideixen perquè gairebé no hi ha seguiment incomplet. Amb aquestes dades, el biaix d'excloure les etiquetes desconegudes no és un problema.
- **Calibratge:** la regressió logística sobreestima el risc entre un 25 % i un 38 % i el random forest l'infraestima entre un 4 % i un 18 %. Encara que els models ordenin bé els discs, cal calibrar-los si les probabilitats s'han d'usar com a tals.
- **Corbes individuals:** amb models separats per horitzó, la corba d'un disc baixa en algun punt per al 5 % dels discs (regressió logística) i per al 27 % (random forest). Un model per franges de temps ho evitaria.

## 5. Aplicació

**Manteniment predictiu:** cada setmana, ordenar els discs per risc de fallar en 30 o 60 dies i substituir o copiar les dades dels primers abans que fallin.

- **Pressupost d'alertes:** l'operador fixa quantes alertes pot assumir (per exemple, l'1 % del parc) i el model decideix quins discs revisar.
- **Comparació amb la regla:** amb unes 3.350 alertes, el random forest detecta la meitat de les fallades. La regla de Backblaze en necessita 26.000 per detectar-ne el 64 %.
- **Corba del risc:** la probabilitat acumulada a 7, 30 i 60 dies permet decidir amb quina urgència s'ha d'actuar per a cada disc.

## 6. Encaix amb el temari

- **Aprenentatge estadístic:** classes molt desequilibrades (corba PR en lloc de l'accuracy), llindars i pressupost d'alertes, validació separada per disc i fuites d'informació (la definició de fallada).
- **Models:** regressió logística, k-NN, Naive Bayes (i per què falla aquí), SVM, arbres i random forests. Permet comparar la importància per impuresa i per permutació, tal com expliquen les diapositives.
- **Probabilitat acumulada:** models per horitzó, comparació amb Kaplan-Meier i el problema de les corbes que baixen.

## 7. Problemes i riscos

- **Definició de fallada:** Backblaze retira proactivament els discs quan els SMART 5, 187, 188, 197 o 198 indiquen una fallada imminent, i al seu sistema de recollida un disc "falla" quan desapareix de la població. Una part de les etiquetes, doncs, surt dels mateixos SMART que el model fa servir.
- **Precisió baixa:** amb una taxa de fallada del 0,1-0,2 %, fins i tot el millor model té una precisió de les alertes inferior al 10 %.
- **SMART heterogenis:** els fabricants reporten diferents SMART i amb significats diferents. El model del disc actua com a variable de confusió.
- **Mida:** les dades d'un sol trimestre ocupen 12 GB. Cal llegir-les fitxer a fitxer i entrenar amb mostres.
- **Un sol trimestre:** l'horitzó màxim és d'uns 60 dies. Per a horitzons més llargs caldria unir diversos trimestres.

## 8. Treballs dels creadors

Backblaze publica estadístiques trimestrals de fallades i articles al seu blog sobre quins SMART indiquen fallades. L'article de ML que hi apareix és d'investigadors externs (Amram et al.), no de l'empresa. No s'ha trobat cap model de predicció publicat per Backblaze, així que el dataset compleix el requisit.

## 9. Què s'ha comprovat i què no

**Comprovat (2026-11-09):**
- Descàrrega i extracció del quart trimestre de 2025.
- Construcció de la taula i recomptes per horitzó.
- Entrenament i avaluació dels sis models, la regla de Backblaze i la prova sense comptadors d'error.
- Comparació amb Kaplan-Meier.
- Definició de fallada i treballs de Backblaze, per cerca web.

**No fet:**
- Ajustar els hiperparàmetres.
- Calibrar les probabilitats, per exemple amb Platt o regressió isotònica.
- Un model per franges de temps.
- Unir diversos trimestres.
- Analitzar per separat les fallades "reactives" (el disc deixa de funcionar) i les "proactives" (retirat per SMART). Les dades no les distingeixen directament.

**Fitxers:**
- Scripts a `backblaze/scripts/`: `consolidate.py`, `build.py`, `explore.py` i `model.py` (amb `uv run python -I`).
- Dades originals a `backblaze/data/q4/` (CSV diaris, 12 GB) i el zip d'1,16 GB.
- Dades consolidades (`consolidate.py data/q4 data 2025Q4`):
  - `backblaze/data/panel/2025Q4.parquet`: una fila per disc i dia (30,9 M files, 889 MB), amb els 63 SMART `_raw` que falten en menys del 99,9 % de files. Hi ha un grup de files per dia. `2025Q4_columns.csv` llista quines columnes s'han conservat.
  - `backblaze/data/drives.parquet`: una fila per disc (341.664), amb primer i últim dia, data de fallada, durada, tipus de censura i hores d'ús.
- `build.py data out/landmarks.parquet` construeix la taula de referència a partir de les dades consolidades.
- Resultats a `backblaze/out/`.

**Fonts:**
- [Backblaze: dades de discs durs](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data)
- [Backblaze: Hard Drive SMART Stats](https://www.backblaze.com/blog/hard-drive-smart-stats/)
- [Backblaze Drive Stats Q3 2025 (definició de fallada)](https://www.backblaze.com/blog/backblaze-drive-stats-for-q3-2025/)
- [Backblaze Drive Stats Q2 2022 (fallades reactives i proactives)](https://www.backblaze.com/blog/backblaze-drive-stats-for-q2-2022/)
- [Backblaze: Using Machine Learning to Predict Hard Drive Failures](https://www.backblaze.com/blog/using-machine-learning-to-predict-hard-drive-failures/)
