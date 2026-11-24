# Preguntes per al professor (classe del 25-11-2026)

Apunteu les respostes a la tasca #41 de Vikunja.

## 1. L'angle propi: de què ens podem refiar

**Context per explicar-li (30 segons):** Backblaze decideix que un disc ha fallat, en part, mirant cinc comptadors SMART d'errors (5, 187, 188, 197 i 198). El nostre model també fa servir aquests comptadors com a variables. Per tant, un model que sembli molt bo pot estar aprenent només la regla de Backblaze, no a anticipar fallades.

**Què proposem:**

- **Escala de variables:** entrenem el mateix model afegint grups de variables un a un (context del disc → altres SMART → absències → comptadors d'error → canvis dels comptadors) i mirem quant guanya cada grup a cada franja de temps.
- **Discs nets:** avaluem a part els discs amb els cinc comptadors a zero, on la regla no pot dir res.
- **Fallades netes:** comptem quantes fallades sense cap error previ detecta el model.

**Pregunta:** *Li sembla una aportació prou bona com a centre del treball?*

## 2. Festes i separació temporal entre entrenament i test

**Context per explicar-li:** entre Thanksgiving i Reis es registren un 57 % menys de fallades. Els discs es deterioren igual (els comptadors d'error continuen pujant), però hi ha menys gent i els discs **es retiren més tard**. Les etiquetes que miren cap a les festes tenen fallades que falten.

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
4. **El test no ha influït en cap decisió:** les festes s'han estudiat només amb dades fins al 4-1-2026, i el test s'executarà una sola vegada (4-12), amb tot congelat.
5. **Per què el test no és novembre–desembre:** seria el període amb menys fallades registrades per les festes, i avaluaríem amb etiquetes incompletes.

**Preguntes:**

- *Li sembla bé tractar les festes tallant les etiquetes d'entrenament el 16-11?*
- *Considera que aquesta separació és suficient perquè el test sigui realment a posteriori, o la faria més gran o d'una altra manera?*
