# Proiect: Analiza Churn la Orange Telecom

**Autor:** Cruceanu Nichita  
**Grupa:** 1098  
**Disciplina:** Pachete Software  

## Descrierea Proiectului
Acest proiect analizează rata de reziliere a contractelor (churn) pentru clienții Orange Telecom, utilizând un set de date real. Obiectivul economic este identificarea factorilor de risc și a profilelor de clienți predispuși să renunțe la serviciile companiei.

Proiectul este structurat în două componente majore, conform cerințelor de la seminar, atingând punctajul maxim pentru funcționalități:
1. **O aplicație web interactivă multi-pagină** dezvoltată în Python (Streamlit).
2. **Un script de analiză statistică și raportare** dezvoltat în SAS.

## Facilități Implementate (Python & SAS)

### Modulul Python
Aplicația Streamlit acoperă 11/11 facilități, dintre care:
* Dezvoltare multi-pagină și widget-uri pentru filtrare.
* Import și procesare avansată cu `pandas` (folosind `index_col=0` la citirea fișierelor CSV și păstrând convenția `_df` pentru seturile de date).
* Analiză cu `scikit-learn` (Clusterizare K-Means pentru clienți) și `statsmodels`.
* Vizualizări grafice complexe utilizând Plotly, Seaborn și Matplotlib.

### Modulul SAS
Codul SAS integrează 10/10 facilități cerute:
1. **Importul** și **concatenarea** datelor externe din CSV.
2. Procesare iterativă utilizând **funcții** și **masive** (`DO`, `ARRAY`, `SUM`, `ROUND`).
3. Crearea de **formate definite de utilizator** (ex. mapare binară pentru ratele de churn).
4. Proceduri specifice **SQL** pentru agregare și combinarea seturilor.
5. Analiză statistică detaliată (`PROC MEANS`) și **Modelare Logistică** (`PROC LOGISTIC`) pentru predicția probabilității de părăsire a rețelei.
6. Generarea de rapoarte și grafice cu `PROC REPORT` și `PROC SGPLOT`.

## Rularea Proiectului
1. **Python:** Din terminal, în directorul aferent fișierelor python, executați comanda: `streamlit run Home.py`.
2. **SAS:** Se încarcă fișierul `.sas` în SAS Studio OnDemand împreună cu seturile de date aferente. Este necesară validarea/actualizarea căilor către fișiere (`FILENAME REF80` și `REF20`) conform locației specifice mediului de rulare de pe server.
