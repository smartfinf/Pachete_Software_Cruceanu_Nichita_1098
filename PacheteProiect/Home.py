import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Orange Telecom - Analiza Churn",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Orange Telecom - Analiza churn-ului clientilor")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Prezentarea companiei Orange S.A.")
    st.markdown("""
    **Orange S.A.** este unul dintre cei mai mari operatori de telecomunicații din lume,
    cu sediul central în Franța și prezent în peste 26 de țări, inclusiv în România.
    Compania oferă o gamă largă de servicii:
    
    - Telefonie mobilă și fixă
    - Internet broadband și 5G
    - Servicii TV și conținut digital
    - Servicii de mesagerie vocală
    - Planuri internaționale pentru convorbiri
    
    În industria telecom, **rata de churn** (procentul de clienți care renunță la servicii)
    este unul dintre cei mai importanți indicatori de performanță. Costurile de achiziție 
    a unui client nou sunt de 5-7 ori mai mari decât costurile de retenție, ceea ce face 
    ca analiza și predicția churn-ului să fie esențiale pentru sustenabilitatea afacerii.
    """)
    
    st.header("Obiectivele proiectului")
    st.markdown("""
    1. **Analiza profilului clienților** Orange Telecom prin metode statistice și vizualizări
    2. **Identificarea segmentelor** de clienți cu risc ridicat de churn (clusterizare)
    3. **Predicția probabilității de churn** prin modele de regresie logistică
    4. **Determinarea factorilor** care influențează valoarea facturii lunare (regresie multiplă)
    5. **Formularea de recomandări** pentru extinderea activității și retenția clienților
    """)

with col2:
    st.header("Sursa datelor")
    st.info("""
    **Dataset:** Orange Telecom Customer Churn Dataset  
    **Sursă:** Kaggle (date publice Orange Telecom)  
    **Înregistrări:** 3.333 clienți  
    **Variabile:** 20 (19 predictori + 1 țintă)  
    **Perioada:** Date operaționale anonimizate
    """)
    
    st.header("Date autor")
    st.success("""
    **Nume:** Cruceanu Nichita  
    **Grupa:** 1098  
    **Facultate:** CSIE, Anul III  
    **Disciplina:** Pachete Software
    """)

st.markdown("---")

st.header("Structura aplicației")

structura = {
    "Pagina": [
        "1. Importul Datelor",
        "2. Analiza Exploratorie",
        "3. Vizualizări Interactive",
        "4. Clusterizare Clienți",
        "5. Predicție Churn (Regresie Logistică)",
        "6. Regresie Multiplă (Statsmodels)"
    ],
    "Descriere": [
        "Importul CSV, tratarea valorilor lipsă, codificare variabile categoriale",
        "Statistici descriptive, gruparea/agregarea datelor, accesare cu loc/iloc",
        "Grafice dinamice cu matplotlib, seaborn și plotly, filtrare interactivă",
        "Segmentarea clienților prin algoritmul K-Means din scikit-learn",
        "Predicția probabilității de churn + metrici (accuracy, precision, recall, AUC)",
        "Modelarea factorilor care influențează factura lunară totală"
    ],
    "Facilități Python": [
        "3, 4, 5",
        "2, 6, 7",
        "2, 8",
        "9",
        "9, 11",
        "10, 11"
    ]
}

st.dataframe(pd.DataFrame(structura), use_container_width=True, hide_index=True)

st.markdown("---")

st.header("Cele 11 facilități Python implementate")

facilitati_col1, facilitati_col2 = st.columns(2)

with facilitati_col1:
    st.markdown("""
    1. **Structură multi-pagină** (Home + 6 pagini)
    2. **Widget-uri** pentru filtrarea interactivă
    3. **Import CSV** în pandas
    4. **Tratarea valorilor lipsă**
    5. **Codificare** variabile categoriale + **scalare**
    6. **GroupBy** și **agregare** în pandas
    """)

with facilitati_col2:
    st.markdown("""
    7. **Accesare cu loc și iloc**
    8. **matplotlib + seaborn + plotly** (grafice dinamice)
    9. **scikit-learn** (KMeans + LogisticRegression)
    10. **statsmodels** (regresie multiplă OLS)
    11. **Afișarea metricilor** (accuracy, precision, recall, F1, AUC)
    """)

st.markdown("---")

