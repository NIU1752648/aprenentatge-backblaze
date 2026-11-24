# Bloc 9. Presentació, teoria, assajos i entrega

**Qui:** Sergi (diapositives i teoria), tots dos (conclusions, assajos, entrega). **Quan:** 7–13-12. **Vikunja:** #60, #61, #62, #157, #63, #64, #65, #66, #67.

## Què demana l'enunciat

- **Presentació de 8 minuts**, repartida a parts iguals (4 minuts cadascú), el 14 o el 16 de desembre. L'**assistència als dos dies és obligatòria**; si falta algú sense justificar, la nota màxima és un 5.
- Companys triats a l'atzar fan preguntes, i forma part de la nota.
- S'avalua: la pregunta, el disseny dels experiments, els resultats i com responem les preguntes.
- **Entrega abans del 13-12 a les 23:59** al Campus Virtual: la presentació en **PDF** i un **zip amb el codi**. La tenim prevista per al 12.

## Esquema proposat (pla §5.3)

| # | Temps | Contingut | Figura clau |
| --- | --- | --- | --- |
| 1 | 0:40 | La pregunta en tres parts (probabilitat, esperança, quins i quan) i l'avís: qui decideix que un disc ha fallat mira els mateixos SMART que el model | La pregunta i el parc en tres xifres |
| 2 | 1:00 | L'etiqueta és una decisió operativa: una regla senzilla ja troba moltes fallades, i les fallades baixen per les festes | Fallades per setmana i cobertura de les regles R1 i R2 |
| 3 | 1:00 | Com evitem enganyar-nos: test en el temps, separació, validació per disc, franges | Línia de temps de la partició i esquema de les tres franges |
| 4 | 1:10 | **Probabilitat:** models, franges, random forest contra gradient boosting, corba acumulada calibrada | Detecció amb l'1 % per model i franja, amb intervals |
| 5 | 1:30 | **De què ens podem refiar:** d'on ve el senyal i amb quina antelació | Escala de variables per franja (E1) i discs nets contra marcats (E2) |
| 6 | 1:00 | **Esperança:** què els passa als discs segons el risc que els donem | Kaplan-Meier per grup de risc fins a 174 dies |
| 7 | 1:10 | **Quins i quan:** pressupost, estalvi, urgència i antelació | Corba de pressupost i corba d'estalvi |
| 8 | 0:30 | Conclusions: una frase per a cada part de la pregunta i els límits | Taula de quatre files |

Al pla, les diapositives 1–3 i 7 eren del paquet A i les 4–6 del B. Amb el repartiment nou, en Sergi les munta i cadascú en presenta 4 minuts; qui presenta cada una es decideix a l'assaig.

**Diapositives de reserva** (per a les preguntes): arbre dibuixat, comparació de validacions (E6), calibratge, cost segons ρ, Naive Bayes, importància per permutació, SHAP, corba d'aprenentatge, validació creuada contra test.

## Feina de Sergi

1. **Estructura i diapositives** (#60, #61): muntar la plantilla i les diapositives a mesura que arribin les figures dels notebooks (bloc 8). Poc text, una idea per diapositiva, la figura gran.
2. **Teoria i preguntes probables** (#157): repassar el pla §4.1 (cada decisió lligada a la diapositiva del temari on surt) i §4.2 (preguntes probables). Escriure una resposta curta per a cada una. Exemples:
   - Per què no feu servir l'accuracy?
   - Per què separeu per disc i no per fila?
   - Per què el test és d'una altra època?
   - Què vol dir que el model està calibrat?
   - El model no està aprenent només la regla de Backblaze?
   - Per què una precisió del 8 % és útil?
3. **Conclusions** (#62, amb Nico): una frase per a cada part de la pregunta (probabilitat, esperança, quins i quan, de què ens podem refiar) i què no podem afirmar.

## Assajos

- **Dc. 9-12:** cronometrat, 8 minuts. Si ens passem, es retalla.
- **Dv. 11-12, amb intercanvi:** cadascú respon les preguntes de la part de l'altre. Si algú no sap respondre alguna cosa, l'altre li explica abans del dia 14.

## Entrega (Nico prepara el zip, #63)

- Netejar el codi, comprovar els notebooks amb `uv run marimo check`, actualitzar el README.
- El zip porta el codi i els resultats petits, **no** les dades (són massa grans; el README explica com descarregar-les i regenerar-les).
- PDF de la presentació.
