# Bloc 2. Hiperparàmetres de gradient boosting i logística

**Qui:** Nico. **Quan:** 25–27-11. **Vikunja:** #47 (E3), #43 (model de franges), #44 (E4), #49 (notebook 03).

## Què són els hiperparàmetres

Són els «botons» d'un model que no s'aprenen de les dades i s'han de triar abans d'entrenar. Per exemple:

- **Regressió logística:** quanta regularització (C). Poca regularització = el model s'ajusta molt a les dades d'entrenament (risc de sobreajust); molta = el model és massa rígid.
- **Gradient boosting:** com de profunds són els arbres, quantes files ha de tenir com a mínim cada fulla, la velocitat d'aprenentatge i quants arbres (aquest últim es tria sol amb early stopping: es deixa d'afegir arbres quan deixen de millorar).

## Com els triem sense enganyar-nos

1. Fem servir **només el desenvolupament** (els 40 dimecres de 2025). El test no es toca.
2. **Validació creuada per disc (`GroupKFold`, 3 grups):** repartim els discs en 3 grups; entrenem amb 2 i avaluem amb el tercer, tres vegades. Un disc mai és alhora a l'entrenament i a l'avaluació, perquè si no el model «el reconeixeria».
3. Per a cada combinació de botons, calculem la **detecció amb l'1 % d'alertes** (la mètrica de selecció) a cada grup i en fem la mitjana.
4. **Regla d'una desviació estàndard:** no triem el millor a seques, sinó el model més senzill que queda a menys d'una desviació estàndard del millor. Els models senzills generalitzen millor i la diferència sovint és soroll.

## Les dues formulacions (E4)

- **Horitzons separats:** un model per a «falla en 7 dies», un per a 30 i un per a 60.
- **Franges:** un model per a cada tram (0–7, 7–30, 30–60 dies), i la probabilitat acumulada surt de combinar-los.

L'script prova totes dues. Ja hem vist a la prova ràpida que amb horitzons separats un 11 % dels discs tenen una corba acumulada que **baixa** (la probabilitat a 60 dies menor que a 30), cosa impossible; amb franges, cap. Aquesta és la comparació E4.

## Les etiquetes amb el tall de festes

Les etiquetes de desenvolupament es tornen a calcular amb informació fins al 16-11-2025 (abans de les festes, quan es registren menys fallades). Fins i tot per saber si un disc segueix viu, només es mira fins al tall. Així l'entrenament no veu res del futur.

## Com s'executa

Al homelab, després del bloc 1, amb la mida triada (unes 5–6 hores):

```
uv run python -I scripts/cv_models.py out/landmarks data out/cv --jobs 8 --neg-drive-frac 0.05
```

(posant a `--neg-drive-frac` la fracció que surti del bloc 1).

## Què produeix

A `out/cv/` (es versiona): `cv_summary.csv` (rendiment de cada configuració), `best_configs.json` (la triada per a cada model i franja), `budget_curve.csv` i informació de l'execució. Les prediccions de validació creuada (grans) queden fora del git i serveixen per al calibratge i el bootstrap (bloc 4).

## Quan està fet

- Configuració triada per a gradient boosting i logística, a cada franja i horitzó.
- Taula comparativa i corbes de validació al notebook `03_models.py`.
- Resultat d'E4 en una frase: franges contra horitzons separats.
