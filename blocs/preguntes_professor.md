# Preguntes per al professor (classe del 25-11-2026)

Apunteu les respostes a la tasca #41 de Vikunja.

## 1. Model base amb els cinc comptadors d'error

**Context per explicar-li (30 segons):** Backblaze retira proactivament discs que mostren errors de lectura i sectors defectuosos als SMART, i aquestes retirades compten com a fallades a les dades. A les nostres dades, els cinc comptadors d'error (SMART 5, 187, 188, 197 i 198) concentren la major part del senyal: un 73 % dels discs que fallen en tenen algun > 0 l'últim dia, contra un 8 % del parc. Per tant, un model que sembli molt bo pot estar aprenent la regla de retirada de Backblaze, no a anticipar fallades.

**Pregunta:** *Li sembla bé que fem servir un model que només faci servir aquests cinc comptadors com a model base, i mesurar quant hi afegeixen la resta de variables (context del disc, altres SMART, absències)? Així podem dir quina part de la predicció és la regla i quina part és anticipació.*

**Si pregunta d'on surt que Backblaze fa servir aquests comptadors:**

- El 2014 Backblaze va escriure que fa servir els SMART 5, 187, 188, 197 i 198 «for determining the failure or potential failure of a hard drive», i que quan el 187 passa de zero programen la substitució del disc ([Hard Drive SMART Stats](https://www.backblaze.com/blog/hard-drive-smart-stats/)).
- El 2022 distingeixen fallades reactives (el disc no respon) i proactives (errors confirmats mirant els SMART); totes dues compten com a fallada ([Drive Stats Q2 2022](https://www.backblaze.com/blog/backblaze-drive-stats-for-q2-2022/)).
- El 2025 diuen que el criteri principal per substituir un disc són els errors de lectura no corregibles, amb un llindar, i que una fallada és un disc amb tiquet de substitució o que desapareix de les dades ([Drive Stats Q3 2025](https://www.backblaze.com/blog/backblaze-drive-stats-for-q3-2025/)).
- No publiquen quina part de les fallades són proactives. Per això no diem que Backblaze faci servir «principalment» aquests cinc comptadors, sinó que són els que concentren el senyal a les dades.

**Complements que ja tenim previstos:** referències sense aprenentatge amb aquests comptadors (algun > 0, algun ha pujat en 7 dies, suma), avaluació a part dels discs amb els cinc comptadors a zero, i de les fallades sense cap error previ.

## 2. Festes i separació temporal entre entrenament i test

**Context per explicar-li:** entre Thanksgiving i Reis es registren un 57 % menys de fallades. Els discs es deterioren igual (els comptadors d'error continuen pujant), però hi ha menys gent i **es registren menys fallades**. Les etiquetes que miren cap a les festes tenen fallades que falten.

**Què proposem:**

```
2025                                                         2026
 F   M   A   M   J   J   A   S   O   N   D   |   G   F   M   A   M   J
 [--------- entrenament (40 dimecres) ---]       [----- test (17) -----]
 5-2                                   5-11      7-1                29-4
                                         |  tall |                 etiquetes
                                     16-11-2025  |                 fins al 28-6
                                                 8-12-2025: el test comença
                                                 a mirar enrere (30 dies)
```

| Bloc | Dates de referència | Quines dades fa servir |
| --- | --- | --- |
| Entrenament | Dimecres del 5-2 al 5-11-2025 | Variables i etiquetes **només fins al 16-11-2025** |
| Separació | Del 12-11 al 31-12-2025 | No es fa servir per a res |
| Test | Dimecres del 7-1 al 29-4-2026 | Variables des del 8-12-2025 (els canvis a 30 dies del primer dimecres) i etiquetes fins al 28-6-2026 |

**Per què el test és realment a posteriori:**

1. **Cap dada de l'entrenament és posterior al 16-11-2025.** Les etiquetes es tallen aquell dia: per a l'horitzó t, només entren els dimecres amb L + t ≤ 16-11-2025. Ni tan sols per saber que un disc segueix viu mirem més enllà del tall.
2. **Cap dada del test és anterior al 8-12-2025.** El primer dimecres de test (7-1-2026) mira 30 dies enrere per calcular els canvis dels comptadors.
3. **Entre les dues hi ha 22 dies sense cap dada compartida** (del 16-11 al 8-12-2025), i 63 dies entre l'últim dimecres d'entrenament i el primer de test, més que l'horitzó més llarg (60 dies). L'entrenament no pot haver vist cap fallada del període de test a través de les etiquetes.
4. **El test no ha influït en cap decisió:** la decisió de les festes es va prendre amb dades fins al 4-1-2026. Després de la classe (25-11) s'han mirat les taxes de fallada del gener–juny de 2026 per comprovar que el ritme torna a pujar (només taxes descriptives, sense cap model), i la decisió no ha canviat. El test s'executarà una sola vegada (4-12), amb tot congelat.
5. **Per què el test no és novembre–desembre:** seria el període amb menys fallades registrades per les festes, i avaluaríem amb etiquetes incompletes.

**Preguntes:**

- *Li sembla bé tractar les festes tallant les etiquetes d'entrenament el 16-11?*
- *Considera que aquesta separació és suficient perquè el test sigui realment a posteriori, o la faria més gran o d'una altra manera?*

**Resposta (25-11-2026) i què n'hem fet:**

- **Submostreig:** no hi ha problema. No és el seu estil, però ens dona millors resultats i el mantenim.
- **Festes:** la caiguda només justifica tractar-les a part si després el ritme torna a pujar. Per comprovar-ho, `notebooks/01_dades.py` ara mostra les dades fins al 30-6-2026, incloent-hi el test:
  - Les pujades de comptadors d'error no baixen en cap moment.
  - Les fallades registrades tornen a pujar just després de Reis: 2,8 per 100.000 discs-dia la primera setmana de gener (0,4–0,8 a final de desembre), 3,2–3,3 al gener–febrer i 4,55 del març al juny, contra 4,14 abans de festes.
  - Per tant, la caiguda va lligada a les festes i mantenim el tall al 16-11 i el split.
  - Matís: les fallades que falten per festes no apareixen més tard (no hi ha cap excés al gener). Les dates de test de gener tenen un 20 % menys de fallades registrades; ho comprovarem traient les dues primeres dates de test i donant les mètriques per data.
