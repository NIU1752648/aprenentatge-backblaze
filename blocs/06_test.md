# Bloc 6. Execució única del test

**Qui:** Nico. **Quan:** divendres 4-12 (fita D3). **Vikunja:** #53.

## Què és el test

Són els 17 dimecres de gener a abril de 2026 (del 7-1 al 29-4), amb les etiquetes fins al 28-6-2026. Fins ara **no l'hem fet servir per a res**: tot s'ha decidit amb el desenvolupament de 2025.

## Per què només una vegada

Si mirem el test, retoquem el model i el tornem a mirar, l'acabem ajustant al test sense voler, i el resultat ja no diu com funcionaria amb dades noves. Per això:

- El 29-11 (D2) es congelen models, variables i hiperparàmetres.
- El 4-12 s'executa el test **una sola vegada**, amb tot congelat.
- S'anota a `context_processament.md` i ja no es torna a executar, encara que els números no ens agradin.

## Què es fa

1. Reentrenar els models triats amb tot el desenvolupament (amb la mostra del bloc 1 i la correcció de probabilitats).
2. Puntuar **tots** els discs de cada dimecres del test, sense mostreig, perquè el pressupost d'alertes es defineix sobre el parc sencer.
3. Calcular les mètriques: detecció amb pressupost, PR-AUC, ROC-AUC, calibratge.
4. Intervals de confiança amb bootstrap per disc (bloc 4) per a les diferències entre models.
5. Resultat extra: només els discs que mai han aparegut a l'entrenament (una avaluació més estricta).
6. Desar les prediccions (fitxer gran, fora del git) per als blocs 5 i 7, i les taules petites al git.

## Què esperem

Probablement el test donarà xifres una mica pitjors que la validació creuada: la validació separa discs però no temps, i el 2026 té discs nous i lots de fallades diferents. Aquesta diferència no és un error: és un resultat que es diu en una frase a la presentació («un model entrenat el 2025 aguanta X al 2026»).

## Quan està fet

Taula de resultats del test amb intervals, prediccions desades, i la data i el commit de l'execució anotats a `context_processament.md`.
