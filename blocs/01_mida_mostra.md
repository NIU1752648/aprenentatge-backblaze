# Bloc 1. Quantes dades fem servir per entrenar

**Qui:** Nico. **Quan:** dimecres 25-11. **Vikunja:** #154 (i decisió 9 del pla).

## El problema

El desenvolupament té 12,8 milions de files (un disc en un dimecres), però només unes 15.000 són fallades a 30 dies. Les altres són discs que no fallen i s'assemblen molt entre elles: el mateix disc surt fins a 40 dimecres seguits amb valors gairebé iguals.

Entrenar amb totes les files és lent (4 minuts per model amb gradient boosting, i k-NN o SVM serien inviables). Per això agafem **totes les fallades i només una part dels discs que no fallen**. La pregunta és: **quina part?** Si n'agafem massa pocs, el model aprèn malament; si n'agafem massa, perdem temps sense guanyar res.

## Què fem

En lloc d'endevinar-ho, ho mesurem amb una **corba d'aprenentatge**:

1. Entrenem el mateix model amb mides de mostra diferents: un 1, 2, 5, 15, 30, 50 i 100 % dels discs que no fallen (sempre amb tots els dimecres d'aquests discs).
2. Per a cada mida, mirem quantes fallades detecta amb l'1 % d'alertes, amb validació creuada per disc (el disc d'avaluació mai s'ha vist a l'entrenament).
3. Dibuixem el rendiment segons la mida. Normalment la corba puja i després s'aplana: el punt on s'aplana és la mida que necessitem.

També provem d'aprimar dimecres (agafar-ne 1 de cada 3 o 1 de cada 10, igual per a fallades i no fallades), per veure si aporten més els discs diferents o els dimecres del mateix disc.

**Compte amb una trampa que vam trobar:** si s'aprimen els dimecres només dels discs que no fallen, el model aprèn a reconèixer la data («aquest dimecres no és a la mostra de negatius, per tant és una fallada») i les probabilitats surten fins a 180 vegades massa altes. Per això l'aprimament és sempre igual per a les dues classes.

**Correcció de les probabilitats:** si entrenem amb un 2 % dels discs sense fallada, el model creu que les fallades són 50 vegades més freqüents del que són. Es corregeix amb una fórmula (p = qπ / (qπ + 1 − q), on π és la fracció conservada). Cada execució comprova que la probabilitat mitjana corregida coincideixi amb la taxa real de fallades.

## Com s'executa

Al homelab, des de `dataset/backblaze/` (unes 5–8 hores):

```
mkdir -p out/lc
nohup uv run python -I scripts/learning_curve.py out/landmarks data out/lc --jobs $(nproc) > out/lc/stdout.txt 2>&1 &
tail -f out/lc/stdout.txt     # per veure el progrés
```

## Què produeix

A `out/lc/` (es versiona): `lc_curve.png` (la corba), `lc_summary.csv`, `lc_recommendation.json` (la mida més petita que queda a una desviació estàndard de la millor) i informació de l'execució.

## Quan està fet

- Tenim la figura i triem la fracció de discs sense fallada (per exemple, «un 5 %»).
- Ho apuntem a la decisió 9 del pla i a Vikunja.
- Llancem el bloc 2 amb aquesta mida.

**Bonus:** és exactament l'exemple de pregunta de l'enunciat («quin impacte té el volum de dades en el rendiment del model?»). La figura pot anar a les diapositives de reserva.
