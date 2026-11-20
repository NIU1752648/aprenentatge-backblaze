# Guió de la presentació

Versió del 20-11-2026. Document de treball per preparar les diapositives (tasques 60, 61 i 62 de Vikunja). Segueix l'esquema de 8 minuts de `pla_projecte.md` (secció 5.3).

**Com fer-lo servir:**

- Cada diapositiva té un missatge en una frase, la figura que el sosté i els números clau, amb el notebook d'on surten.
- Els números es copien dels notebooks, no es recalculen aquí. Si un notebook canvia, s'actualitza la xifra i la data.
- Etiquetes d'estat:
  - **[fet]**: resultat definitiu.
  - **[preliminar]**: només amb dades de desenvolupament i models simples. Pot canviar amb el `build.py` nou i els models finals.
  - **[pendent]**: encara no hi ha resultat.
- Figures: `dataset/backblaze/out/figures/`.

## Estat general

| # | Temps | Qui | Diapositiva | Estat |
| --- | --- | --- | --- | --- |
| 1 | 0:40 | A | La pregunta i el parc | Xifres del parc [fet] |
| 2 | 1:00 | A | L'etiqueta és una decisió operativa | [fet] |
| 3 | 1:00 | A | Com evitem enganyar-nos | Disseny decidit; falta la figura de la partició |
| 4 | 1:10 | B | Probabilitat | [pendent] (fita D2) |
| 5 | 1:30 | B | De què ens podem refiar | [preliminar] |
| 6 | 1:00 | B | Esperança | [pendent] (fita D3) |
| 7 | 1:10 | A | Quins i quan | Una peça [preliminar]; la resta [pendent] |
| 8 | 0:30 | A i B | Conclusions | [pendent] |

---

## 1 · La pregunta i el parc (A, 0:40)

**Missatge.** Volem saber quina probabilitat hi ha que un disc falli en els propers t dies, quina esperança li podem donar i quins discs s'han de substituir i quan. Un avís des del principi: qui decideix que un disc ha fallat mira els mateixos SMART que el model.

**La pregunta** (tal com es va enviar al professor):

> A partir de l'estat SMART d'un disc en un moment donat, quina probabilitat hi ha que falli en els propers t dies, quina esperança podem donar a aquests discs i com podem fer servir aquesta informació per saber quins discs s'han de substituir i quan?

**El parc en tres xifres** [fet] (`context_processament.md`, `drives.parquet`):

- **372.456 discs** diferents al llarg de **546 dies** (1-1-2025 – 30-6-2026), de 305.000 a 355.000 actius alhora.
- **6.861 fallades**: entre un 0,02 % i un 0,2 % dels discs fallen en cada horitzó. Per això no fem servir l'accuracy.
- Dades públiques de Backblaze Drive Stats, sis trimestres (Q1 2025 – Q2 2026).

**Figura.** Títol, la pregunta i les tres xifres. No cal gràfic.

---

## 2 · L'etiqueta és una decisió operativa (A, 1:00)

**Missatge.** «Fallat» vol dir «Backblaze l'ha retirat», no «s'ha trencat». Per festes es retiren menys de la meitat de discs, mentre els discs es deterioren igual.

**Figura principal:** `nadal_setmanal.png` (notebook `01_dades.py`). Tres panells setmanals:

- fallades, que cauen;
- pujades de comptadors d'error, que es mantenen o pugen;
- discs que surten sense fallar.

**Números** [fet] (`01_dades.py`, dades fins al 4-1-2026):

- Del 17-11-2025 al 4-1-2026: **−57 %** de fallades registrades per disc respecte de les set setmanes anteriors (1,80 contra 4,14 per 100.000 discs-dia). Les dues últimes setmanes de desembre, **−90 %**.
- En aquest mateix període, les pujades de comptadors d'error **augmenten un 10 %**, i les aparicions noves d'errors no canvien.
- La caiguda comença la setmana de Thanksgiving, no per Nadal.
- Dia de la setmana: es registren un **35 % menys** de fallades en diumenge i dilluns. Els comptadors no varien segons el dia.

**Frase per dir-ho:** «Un disc no sap quin dia és. El tècnic que el retira, sí.»

**Segona peça (del pla):** una regla senzilla ja troba moltes fallades. Amb l'1 % d'alertes, la regla del tècnic (comptadors que han pujat) captura un **46 %** de les fallades a 30 dies [preliminar, `01b_etiquetes.py`, E1].

**Pendent / decidir:**

- El pla demanava la figura d'octubre de 2025 a març de 2026. Els mesos de 2026 són del període de test i no es poden mirar fins a l'execució única del test (fita D3). Fins llavors, la figura acaba el 4-1-2026.
- Cobertura de R1 i R2 sobre el parc (`build.py` nou).

**Reserva:** `dia_setmana.png`, `nadal_retard.png`.

---

## 3 · Com evitem enganyar-nos (A, 1:00)

**Missatge.** Avaluem com si fos el futur: test posterior en el temps, separació de 60 dies, discs separats entre entrenament i avaluació, i etiquetes de festes fora de l'entrenament.

**Disseny** (pla, secció 1.5; decisió de Nadal pendent de confirmar a Vikunja, tasca 70):

- Desenvolupament: dates del 5-2 al 5-11-2025. Test: del 7-1 al 29-4-2026. No es toca fins al final.
- **Censura per festes:** per a l'horitzó t, només s'entrenen les dates amb L + t ≤ 16-11-2025. Es perden 0, 3 i 7 de les 40 dates a 7, 30 i 60 dies.
- Validació creuada per disc (`GroupKFold`).

**Per què separem per disc** [preliminar, `01b_etiquetes.py`, nota de la secció 3]: cada disc apareix a unes 40 dates gairebé iguals. Un gradient boosting poc regularitzat memoritzava els discs, i amb discs nous preveia **2,2 vegades** les fallades reals. Regularitzant-lo, millora (PR-AUC de 0,115 a 0,172).

**Figura.** [pendent] Línia de temps de la partició amb la zona de festes marcada, i l'esquema d'una data amb les tres franges.

**Reserva:** `nadal_etiquetes.png` (quines etiquetes de desenvolupament toquen les festes).

---

## 4 · Probabilitat (B, 1:10)

**Missatge.** [pendent] Set models del curs, tres franges, corba acumulada monòtona i calibrada.

**Figura.** [pendent] Barres de detecció amb l'1 % per model i franja, amb intervals; corbes acumulades de tres discs.

**Ja sabem** [preliminar, `01b_etiquetes.py`]:

- Entrenar amb etiquetes de festes **no canvia quins discs avisem** (diferència de recall: −0,002 [−0,006, 0,001]). Però rebaixa un 4 % la probabilitat. La censura protegeix sobretot la calibració.
- Amb una mostra del 5 % de discs sans, el gradient boosting sobreestima el risc un 25–30 % en discs nous. Cal recalibrar (tasca 45) o fer servir una mostra de negatius més gran.

---

## 5 · De què ens podem refiar (B, 1:30)

**Missatge.** Bona part del que encerta el model és la mateixa informació que fa servir Backblaze per retirar discs. Les fallades que sorprenen el tècnic hi són, però costa arribar-hi.

**Números** [preliminar, `01b_etiquetes.py`, E1, t = 30, 1 % d'alertes, validació creuada per disc]:

| | Totes les fallades | Fallades brutes (amb errors) | Fallades netes (sense errors, 24 %) |
| --- | --- | --- | --- |
| Regla del tècnic | 46 % | 60 % | 0,7 % |
| Gradient boosting | 52 % | 67 % | 3 % |
| Gradient boosting sense els 5 comptadors | 15 % | 12 % | 27 % |

- Millora del model sobre la regla: +6 punts de recall [4,9, 6,8].
- Sense comptadors, el model troba fallades netes que el model complet no prioritza.

**Figura.** [pendent] Escala de variables × franja (E1) i discs nets contra marcats (E2), amb els models finals. La taula de sobre ja és un primer esborrany.

**Límit que cal dir:** «net» i «brut» són aproximacions a «reactiu» i «proactiu». Les dades no diuen per què es va retirar cada disc.

---

## 6 · Esperança (B, 1:00)

**Missatge.** [pendent] Què els passa als discs segons el risc que els donem.

**Figura.** [pendent] Supervivència observada per grup de risc al test, fins a 174 dies (E7).

---

## 7 · Quins i quan (A, 1:10)

**Missatge.** [pendent] Amb un pressupost d'alertes, quantes fallades avisem i amb quin marge.

**Ja sabem** [preliminar, `01b_etiquetes.py`, E2]:

- Molts falsos positius són avisos massa d'hora. Un **9 %** dels FP a 30 dies fallen entre 30 i 90 dies després (contra un 0,15 % dels discs sense alerta), i un **96 %** ja tenen errors.
- Si els comptem com a encerts, la precisió de les alertes passa del **6 % al 14 %**.
- La matriu de confusió d'una data típica: unes 3.160 alertes, 189 encerts, i 175 fallades que s'escapen.
- Lliga amb les classes d'urgència: un disc amb risc alt però tardà és una alerta «no urgent».

**Figura.** [pendent] Corba detecció–pressupost contra R2 i R3, i histograma de l'antelació (E5).

---

## 8 · Conclusions (A i B, 0:30)

[pendent] Taula de quatre files: probabilitat, esperança, quins i quan, i què no podem afirmar.

**Esborrany de la fila «què no podem afirmar»** [preliminar]: el model aprèn quan Backblaze retirarà un disc, no quan es trencarà. Per ordenar discs, l'error de l'etiqueta gairebé no afecta. Per donar probabilitats, sí, i per això traiem les etiquetes de festes i recalibrem.

---

## Diapositives de reserva

| Figura | Respon a | Font |
| --- | --- | --- |
| `dia_setmana.png` | «Com sabeu que l'etiqueta és humana?» | `01_dades.py` |
| `nadal_retard.png` | «És retard en la retirada o menys fallades?» | `01_dades.py`, N3 |
| `nadal_etiquetes.png` | «Quines etiquetes toquen les festes?» | `01_dades.py`, N4 |
| `nadal_variables.png` | «Les dates de test de gener veuen variables diferents?» | `01_dades.py`, N5 |
| `etiqueta_festes_prevista.png` | «Quantes fallades falten a l'etiqueta?» El model n'estima un 32 % a les 7 dates afectades | `01b_etiquetes.py`, E3 |
| `etiqueta_soroll.png` | «Quant error d'etiqueta aguanta el model?» El rànquing no es mou ni amagant la meitat de les fallades | `01b_etiquetes.py`, E4 |

La resta de reserves del pla (arbre dibuixat, E6, calibratge, cost segons ρ, Naive Bayes, importància per permutació) estan pendents.

## Preguntes noves per a la secció 4.2 del pla

| Pregunta | Resposta breu |
| --- | --- |
| Si l'etiqueta té errors, el model no els arrossega al futur? | Traiem els errors coneguts (festes) i mesurem la resta. Per ordenar discs, gairebé no afecta (E3, E4). Per a les probabilitats, sí: per això traiem les etiquetes de festes i recalibrem |
| Què és un fals positiu aquí? | Una alerta per a un disc que no consta com a fallat en t dies. Molts són avisos anticipats: un 9 % fallen entre 30 i 90 dies després |
| Per què les fallades cauen per Nadal? | Per festes es retiren menys discs. Els comptadors d'error no baixen. L'efecte comença per Thanksgiving |
| Què va passar el 12-5-2025? | Una fallada de la recollida de dades: un 29 % dels discs van faltar un sol dia. No es tracta com a absència real |
