# Bloc 10. Cerca bibliogràfica i seguiment amb el professor

**Qui:** Sergi. **Quan:** aquesta setmana (abans del 29-11). **Vikunja:** #40 (bibliografia), #41 (professor).

## Cerca bibliogràfica

**Per què:** un dels requisits del professor és que **els creadors del dataset (Backblaze) no hagin publicat la mateixa anàlisi**. També ens va bé saber què han fet altres per citar-ho en una frase i explicar què fem diferent.

**Què buscar:**

1. **Blog de Backblaze (Drive Stats):** publiquen informes trimestrals i anuals. Sobretot taxes de fallada anuals per model (AFR) i, alguna vegada, articles sobre SMART i sobre vida per edat. Comprovar:
   - Han publicat models per predir fallades a 7, 30 o 60 dies?
   - Han estudiat quina part de la predicció és la seva pròpia regla (els comptadors 5, 187, 188, 197 i 198)?
   - Publiquen corbes de vida per edat o per model? (Si és així, la nostra «taxa per model i edat» és només una referència, no un resultat.)
2. **Articles acadèmics** sobre predicció de fallades de discs amb les dades de Backblaze (n'hi ha molts). Buscar a Google Scholar, per exemple: «Backblaze hard drive failure prediction SMART», «disk failure prediction machine learning». Mirar:
   - Quin horitzó fan servir i com separen entrenament i test (molts barregen files del mateix disc, que és justament l'error que evitem).
   - Si algú parla del problema que l'etiqueta depèn dels mateixos SMART.

**Què lliurar:** una nota curta (un `.md` a `docs/` o un comentari a Vikunja #40) amb 3–6 referències, una línia per a cada una («què fan» i «en què ens diferenciem»), i la conclusió: «Backblaze no ha publicat aquesta anàlisi» (o el contrari, i què canviem).

## Seguiment amb el professor

**Què ensenyar-li a la propera classe** (#41):

1. **L'angle propi:** com que Backblaze decideix les fallades en part amb els mateixos comptadors que fa servir el model, mesurem quina part és la regla i quina és anticipació (bloc 5).
2. **La definició d'esperança:** supervivència observada amb Kaplan-Meier per grup de risc, fins a 174 dies, sense extrapolar (bloc 7).
3. Opcional: la corba d'estalvi segons el pressupost (bloc 7), si ja la tenim amb dades reals.

**Què preguntar-li:**

- Li sembla bé com responem «esperança»?
- Li sembla prou l'angle propi com a aportació?
- Alguna preferència sobre quins models ensenyar a la presentació?

**Què lliurar:** un comentari a Vikunja #41 amb el que hagi dit, i avisar en Nico si canvia alguna cosa del pla.
