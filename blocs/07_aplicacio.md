# Bloc 7. Aplicació: quins discs i quan, i esperança

**Qui:** Nico (E5: alertes, urgència, antelació, estalvi), Sergi (Kaplan-Meier). **Quan:** 4–6-12 (codi de Kaplan-Meier abans del 29-11). **Vikunja:** #51 (E5), #156 i #52 (Kaplan-Meier).

Aquest bloc respon dues parts de la pregunta: «com fem servir la informació per saber quins discs s'han de substituir i quan» i «quina esperança podem donar a aquests discs». Tot es fa amb les prediccions del test (bloc 6).

## E5. Quins discs: pressupost i estalvi (Nico)

**Pressupost d'alertes:** cada setmana l'equip només pot revisar uns quants discs. Ordenem els discs per risc i marquem el k % de dalt.

**Figura 1, corba de pressupost:** per a cada k (del 0,01 % al 10 %), quin percentatge de les fallades reals hi havia entre els marcats. Marquem el 0,1 % (~350 discs per setmana) i l'1 % (~3.500).

**Figura 2, estalvi segons el pressupost:** posem preu als errors.

- Revisar o canviar un disc avisat costa **1**.
- Una fallada no avisada costa **ρ** (5, 20 o 100 vegades més).
- **Estalvi = (ρ − 1) × fallades avisades − falses alarmes.** Cada fallada avisada estalvia ρ − 1 (passa de costar ρ a costar 1); cada falsa alarma costa 1.
- Ho mostrem com a percentatge de l'estalvi màxim (avisar totes les fallades i cap disc sa).

La corba puja, arriba a un màxim i baixa quan les falses alarmes costen més del que estalvien. **El màxim és el pressupost òptim.** A la prova: amb ρ = 20, cal revisar un 0,36 % del parc cada setmana; amb un 1 % fix, si ρ = 5 es perden diners.

## E5. Quan: urgència i antelació (Nico)

- **Classes d'urgència** amb el model de franges: *urgent* si la probabilitat de fallar en 7 dies és alta (copiar dades aquesta setmana), *programar* si ho és a 30 dies (demanar recanvi), *vigilar* si ho és a 60.
- **Antelació:** dies entre el primer avís i la fallada. Quantes fallades s'avisen amb almenys 7 dies de marge?

## E7. Esperança amb Kaplan-Meier (Sergi)

**Què volem dir amb «esperança»** (decisió 15): no diem «a aquest disc li queden X anys» (no ho podem saber sense inventar), sinó **com sobreviuen de veritat els discs que el model posa en risc alt, comparats amb els de risc baix**.

**Com:**

1. Al primer dimecres del test (7-1-2026), el model puntua tots els discs. Els repartim en quatre grups: el 0,1 % amb més risc, del 0,1 % a l'1 %, de l'1 % al 10 %, i la resta.
2. Seguim cada disc fins al final de les dades (30-6-2026, 174 dies): o falla (i sabem el dia), o surt de les dades sense fallar, o arriba viu al final.
3. Per a cada grup, dibuixem la **corba de Kaplan-Meier**: el percentatge de discs que segueixen vius cada dia.

**Per què Kaplan-Meier i no comptar a mà:** alguns discs surten de les dades sense fallar (es retiren, canvien de centre...). No sabem si haurien fallat després. Kaplan-Meier els compta com a vius mentre els veiem i després els treu del càlcul, sense suposar res. La fórmula: cada dia amb fallades, S(t) = S(t anterior) × (1 − fallades del dia / discs encara en observació aquell dia).

**Feina abans del test (#156):** escriure el codi i provar-lo ara amb el desenvolupament. Com que encara no hi ha prediccions, els grups es poden fer amb una referència (per exemple R3, la suma de comptadors, bloc 4). Les dades necessàries:

- `drives.parquet`: `fail_date`, `last_seen` i `censoring` de cada disc.
- El dimecres de partida: un fitxer de `out/landmarks/`.
- Temps de cada disc = dies des del dimecres fins a `fail_date` (si falla) o fins a `last_seen` (si no). Esdeveniment = 1 si falla.

Quan arribin les prediccions del test (bloc 6), només cal canviar la puntuació que fa els grups.

**Què es diu a la presentació:** per a cada grup, quin percentatge segueix viu a 60, 120 i 174 dies. Per exemple: «dels discs del grup de risc més alt, en segueix viu un X % al cap de 60 dies; dels del grup més baix, el 99,8 %».

## Quan està fet

Les dues figures d'E5, la taula d'urgència i antelació, i les corbes de Kaplan-Meier per grup, al notebook `05_aplicacio.py`.
