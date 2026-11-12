# Projecte: datasets per estudiar el risc en funció del temps

Vint datasets de lliure accés que donen el temps fins a un esdeveniment. Permeten crear l'objectiu fixant un horitzó `t` ("l'esdeveniment passa abans de `t`?") i entrenar models per a diversos horitzons o franges de temps, tal com va proposar el professor.

**Recomanació:**
- **AIDS 175 (#1)** per començar ja: està descarregat i comprovat.
- **NHANES amb mortalitat (#2)** si voleu l'aplicació més clara (cribratge de risc).
- **Backblaze (#14)** si preferiu una aplicació fora de la salut (manteniment predictiu).

## Requisits

1. **Temps fins a l'esdeveniment:** dates o durades exactes, no franges fixes. El Diabetes 130 queda fora perquè només dona `<30`, `>30` o `NO`.
2. **Accés lliure:** descàrrega directa o amb un compte gratuït, sense credencials, acords signats ni sol·licituds. Això deixa fora MIMIC-IV.
3. **Sense habitatge**, com en les llistes anteriors.
4. **Els creadors no han publicat el mateix estudi:** els treballs d'altres investigadors no compten.
5. **Aplicació clara:** una decisió que el model ajudaria a prendre.

## Com s'ha comprovat

- **Comprovat (2026-11-09):** existència, accés i variables de temps de cada dataset, amb cerques web. Les mides només hi són quan una font les dona.
- **AIDS 175:** descarregat i analitzat. Té 2.139 pacients i 521 esdeveniments, amb seguiment fins a 1.231 dies, i serveix per a horitzons de fins a uns 900 dies.
- **No provat:** la resta no s'ha descarregat. Les notes d'"Atenció" surten de les cerques i de coneixement general.
- **Treballs dels creadors:** s'ha marcat com a risc quan s'ha trobat un treball semblant. "No comprovat" vol dir que no s'ha buscat expressament.

## Com construir l'objectiu

- **Un model per horitzó:** per a cada `t` (per exemple 30, 90 i 180 dies), l'objectiu és 1 si l'esdeveniment passa abans de `t`. La sortida dels models per als diferents `t` dona la probabilitat acumulada.
- **Seguiment incomplet:** si el seguiment d'un cas acaba abans de `t` sense esdeveniment, la seva etiqueta és desconeguda i s'exclou d'aquell horitzó. Aquest grup creix amb `t` i pot esbiaixar els resultats, cosa que convé discutir a l'informe.
- **Franges de temps:** un sol model prediu la probabilitat d'esdeveniment en cada franja, amb la franja com a variable d'entrada. La probabilitat acumulada s'obté combinant-les i no pot baixar amb el temps, a diferència dels models separats per horitzó.
- **Referència:** la corba de Kaplan-Meier permet comprovar que la mitjana de les probabilitats predites té sentit.
- **Models:** regressió logística, k-NN, Naive Bayes, SVM, arbres de decisió i random forests serveixen per a tots dos enfocaments.

## Salut

| # | Dataset | Fila i temps | Esdeveniment | Aplicació | Atenció |
| --- | --- | --- | --- | --- | --- |
| 1 | [AIDS Clinical Trials Group Study 175](https://archive.ics.uci.edu/dataset/890/aids+clinical+trials+group+study+175) (UCI) | Pacient; dies de seguiment (`time`) | Progressió (caiguda de CD4, sida o mort) | Decidir amb quina freqüència fer seguiment a cada pacient des de l'inici del tractament | `cd420`, `cd820` i `offtrt` es coneixen després de l'inici i filtrarien el resultat |
| 2 | [NHANES](https://wwwn.cdc.gov/nchs/nhanes/tutorials/datasets.aspx) amb [dades de mortalitat](https://www.cdc.gov/nchs/linked-data/mortality-files/) (CDC) | Adult enquestat; mesos fins a la mort o fins al final del seguiment | Mort | Cribratge del risc de mort a 5, 10 o 15 anys a partir d'exàmens mèdics | Cal unir fitxers per identificador. Les fonts no coincideixen sobre fins a quin any arriba el seguiment |
| 3 | [COVID-19 a Mèxic](https://www.gob.mx/salud/documentos/datos-abiertos-152127) (Secretaria de Salut) | Cas; dates d'inici de símptomes, ingrés i mort | Mort | Triatge en ingressar: risc de morir en 7, 14 o 30 dies | Molt gran. Els vius no tenen data de final de seguiment, cal fixar-la |
| 4 | [`flchain`](https://www.rdocumentation.org/packages/survival/versions/3.8-6/topics/flchain) (paquet R `survival`) | Persona (7.874); dies fins a la mort | Mort | Valor d'una analítica de sang per estimar el risc a llarg termini | Mostra d'un estudi d'un sol comtat dels EUA |
| 5 | [`mgus2`](https://therneau.r-universe.dev/survival) (paquet R `survival`) | Pacient (1.384 segons una còpia); mesos | Progressió a càncer o mort | Decidir a qui fer un seguiment més intens | Dos esdeveniments que competeixen entre ells |
| 6 | [`colon`](https://r-packages.io/packages/survival/colon) (paquet R `survival`) | Pacient (929); dies | Recaiguda o mort | Risc de recaiguda després de la cirurgia | L'original té dues files per pacient, una per esdeveniment |
| 7 | [METABRIC](https://datacatalog.mskcc.org/dataset/11457) (cBioPortal) | Pacient amb càncer de mama (2.509); mesos fins a la mort | Mort | Pronòstic a 5 i 10 anys a partir de dades clíniques | Les dades es van fer servir en un concurs de pronòstic el 2012, cal comprovar si hi van participar els creadors |
| 8 | [TCGA](https://bioconductor.statistik.tu-dortmund.de/packages/3.22/bioc/vignettes/TCGAbiolinks/inst/doc/clinical.html) (portal GDC) | Pacient amb càncer; dies fins a la mort o l'últim seguiment | Mort | Pronòstic per tipus de càncer | Hi ha incoherències conegudes entre taules |

## Economia i societat

| # | Dataset | Fila i temps | Esdeveniment | Aplicació | Atenció |
| --- | --- | --- | --- | --- | --- |
| 9 | [`UnempDur`](https://search.r-project.org/CRAN/refmans/Ecdat/html/UnempDur.html) (paquet R `Ecdat`) | Període d'atur (3.343); durada en quinzenes | Tornar a treballar, a temps complet o parcial | Detectar qui pot quedar en atur de llarga durada | Dades de 1986-1992 |
| 10 | [Rossi](https://search.r-project.org/CRAN/refmans/carData/html/Rossi.html) (paquet R `carData`) | Persona alliberada de presó (432); setmanes fins a la detenció | Nova detenció | Avaluar un programa d'ajuda econòmica | Petit, i el seguiment s'acaba a les 52 setmanes |
| 11 | [NSFG 2022-2023](https://cdc.gov/nchs/data/nsfg/guidefaqs/NSFG-2022-2023-FAQ.pdf) (CDC) | Persona enquestada; dates dels esdeveniments | Per exemple, primer matrimoni o primera convivència | Estudis demogràfics | Cal confirmar quines variables de dates són públiques, i tenir en compte els pesos de l'enquesta |
| 12 | [Lending Club](https://zenodo.org/records/11295916) (Kaggle i Zenodo) | Préstec (2007-2018); dates de concessió i últim pagament | Impagament | Risc d'impagament als 6, 12 o 24 mesos | Els préstecs encara vius tenen el seguiment incomplet. La llicència de les còpies no és clara |
| 13 | [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (UCI) | Compra (1.067.371, de 5.942 clients); dates | Nova compra, o deixar de comprar | Quins clients cal intentar retenir i quan | Cal agrupar per client, i 243.007 files no tenen client. Els creadors van fer segmentació de clients i previsió de rendibilitat, no aquest estudi |

## Operacions i manteniment

| # | Dataset | Fila i temps | Esdeveniment | Aplicació | Atenció |
| --- | --- | --- | --- | --- | --- |
| 14 | [Backblaze](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data) | Disc dur i dia; dades SMART diàries | Fallada del disc | Quins discs cal substituir abans de 7, 30 o 90 dies | Molt gran i molt desequilibrat. Cal citar Backblaze i no es poden vendre les dades |
| 15 | [NASA C-MAPSS](https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data) | Motor i cicle; sensors | Fallada del motor | Planificar el manteniment d'avions | Dades simulades. Es van publicar per a un concurs de predicció |
| 16 | [Sol·licituds 311 de Nova York](https://catalog.data.gov/dataset/311-service-requests-from-2010-to-present) | Sol·licitud de servei; data de creació i de tancament | Resolució | Quines sol·licituds no es resoldran en 1, 7 o 30 dies | Enorme, i dividit en dos datasets (2010-2019 i 2020-actualitat) |
| 17 | [AACT (ClinicalTrials.gov)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3306288) | Assaig clínic; dates d'inici, finalització i aturada | Aturada anticipada o finalització | Detectar quins assajos poden aturar-se abans d'hora | Dates declarades pels investigadors. Finalitzar i aturar-se són esdeveniments que competeixen |

## Amb risc d'encavalcament

| # | Dataset | Fila i temps | Esdeveniment | Risc |
| --- | --- | --- | --- | --- |
| 18 | [COMPAS (ProPublica)](https://dlab-berkeley.github.io/fairML/bias-in-the-algorithm.html) | Persona (uns 13.000); dies fins a la reincidència | Reincidència | ProPublica va fer un model de supervivència sobre la puntuació de risc. Tema delicat d'equitat |
| 19 | [Insuficiència cardíaca](https://rdrr.io/cran/CardioDataSets/man/cardiac_failure_df.html) (UCI) | Pacient (299); dies de seguiment | Mort | UCI el vincula a un article que ja prediu la supervivència amb ML (Chicco i Jurman, 2020). És molt petit |
| 20 | [Rotació de personal (Babushkin)](https://www.aihr.com/blog/applying-survival-analysis-reduce-employee-turnover-practical-case/) (Kaggle) | Treballador (1.129 en una còpia); temps a l'empresa | Marxa de l'empresa | El creador va publicar una anàlisi de supervivència amb aquestes dades |

## Descartats

| Dataset | Motiu |
| --- | --- |
| MIMIC-IV i eICU | Calen credencials de PhysioNet, que triguen 1-2 setmanes, i les dades no es poden compartir dins l'equip |
| SEER, Framingham (BioLINCC), UK Biobank | Cal un acord signat o una sol·licitud |
| SUPPORT2 | Els creadors van publicar un model de pronòstic de supervivència (Knaus, 1995) |
| Cirrosi / PBC (Mayo) | Els creadors van publicar un model de risc de supervivència |
| OULAD (Open University) | Els creadors ja fan prediccions d'abandonament (OU Analyse) |
| Freddie Mac i Fannie Mae | Préstecs hipotecaris, és a dir, habitatge |
| Diabetes 130-US Hospitals | Només dona franges fixes de readmissió (`<30`, `>30`, `NO`) |
| Inspeccions alimentàries de Chicago | L'ajuntament va construir el seu propi model de predicció |

## Millors opcions segons l'objectiu

- **Començar ja:** AIDS 175 (#1).
- **Aplicació més clara:** NHANES amb mortalitat (#2), per al cribratge del risc a diversos anys vista.
- **Més a prop de la readmissió:** COVID-19 a Mèxic (#3), per al risc des de l'ingrés hospitalari.
- **Aplicació fora de la salut:** Backblaze (#14), per al manteniment predictiu.
- **Treballar esdeveniments que competeixen:** `mgus2` (#5) o AACT (#17).

## Següent pas

Triar dos o tres datasets i comprovar que el seguiment permet els horitzons que voleu. Després, entrenar models de referència per a diversos horitzons i comparar la probabilitat acumulada amb la corba de Kaplan-Meier.
