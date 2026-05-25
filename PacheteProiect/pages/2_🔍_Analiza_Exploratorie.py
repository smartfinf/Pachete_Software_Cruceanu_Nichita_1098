import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Analiza Exploratorie", layout="wide")

st.title("Analiza Exploratorie a Datelor (EDA)")
st.markdown("---")

if 'df_clean' not in st.session_state:
    project_root = Path(__file__).resolve().parents[1]
    combined_path = project_root / "data" / "orange_telecom_churn.csv"
    if combined_path.exists():
        df = pd.read_csv(combined_path)
    else:
        part_80 = project_root / "churn-bigml-80.csv"
        part_20 = project_root / "churn-bigml-20.csv"
        if part_80.exists() and part_20.exists():
            df = pd.concat([pd.read_csv(part_80), pd.read_csv(part_20)], ignore_index=True)
        else:
            st.error("Nu am găsit dataset-ul local. Pune `churn-bigml-80.csv` și `churn-bigml-20.csv` în folderul proiectului.")
            st.stop()

    if 'Churn' in df.columns and df['Churn'].dtype == object:
        churn_map = {'true': True, 'false': False}
        df['Churn'] = df['Churn'].astype(str).str.strip().str.lower().map(churn_map)

    st.session_state['df_clean'] = df

df = st.session_state['df_clean'].copy()

st.header("Filtrare interactivă a datelor")

st.markdown("""
**Definirea problemei:** Permitem utilizatorului (managerul de business) să exploreze 
interactiv segmente specifice din baza de clienți, filtrând după plan internațional, 
plan mesagerie vocală, durata abonamentului și numărul de apeluri la serviciul clienți.
""")

col_f1, col_f2, col_f3, col_f4 = st.columns(4)

with col_f1:
    intl_plan = st.multiselect(
        "Plan internațional",
        options=df['International plan'].unique(),
        default=df['International plan'].unique()
    )

with col_f2:
    vmail_plan = st.multiselect(
        "Plan mesagerie vocală",
        options=df['Voice mail plan'].unique(),
        default=df['Voice mail plan'].unique()
    )

with col_f3:
    account_range = st.slider(
        "Durata cont (zile)",
        int(df['Account length'].min()),
        int(df['Account length'].max()),
        (int(df['Account length'].min()), int(df['Account length'].max()))
    )

with col_f4:
    cs_calls_max = st.slider(
        "Max apeluri Serviciu Clienți",
        int(df['Customer service calls'].min()),
        int(df['Customer service calls'].max()),
        int(df['Customer service calls'].max())
    )

df_filtered = df[
    (df['International plan'].isin(intl_plan)) &
    (df['Voice mail plan'].isin(vmail_plan)) &
    (df['Account length'] >= account_range[0]) &
    (df['Account length'] <= account_range[1]) &
    (df['Customer service calls'] <= cs_calls_max)
]

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Clienți filtrați", f"{len(df_filtered):,}")
col_m2.metric("Rata churn filtrată", f"{df_filtered['Churn'].mean()*100:.2f}%")
col_m3.metric("Factură medie zi ($)", f"{df_filtered['Total day charge'].mean():.2f}")

st.markdown("---")

st.header("Gruparea și agregarea datelor (pandas groupby)")

st.markdown("""
**Definirea problemei:** Pentru a înțelege ce caracteristici sunt asociate cu churn-ul, 
grupăm clienții pe diverse dimensiuni și calculăm metrici relevante: rata medie de churn, 
factura medie, vechimea medie.
""")

tab1, tab2, tab3, tab4 = st.tabs([
    "Churn după Planul Internațional",
    "Churn după Planul Voicemail",
    "Churn după Apeluri Serviciu Clienți",
    "Top 10 State după rata Churn"
])

with tab1:
    g1 = df.groupby('International plan').agg(
        Numar_clienti=('Churn', 'count'),
        Rata_churn=('Churn', 'mean'),
        Cost_mediu_zi=('Total day charge', 'mean'),
        Vechime_medie=('Account length', 'mean')
    ).round(3)
    g1['Rata_churn'] = (g1['Rata_churn'] * 100).round(2).astype(str) + ' %'
    st.dataframe(g1, use_container_width=True)
    st.info("""
    **Interpretare:** Clienții cu plan internațional au o rată de churn semnificativ mai mare 
    decât cei fără. Aceasta sugerează că planul internațional nu satisface complet nevoile 
    clienților, posibil din cauza tarifelor sau a calității acoperirii internaționale.
    """)

with tab2:
    g2 = df.groupby('Voice mail plan').agg(
        Numar_clienti=('Churn', 'count'),
        Rata_churn=('Churn', 'mean'),
        Mesaje_medii=('Number vmail messages', 'mean'),
        Vechime_medie=('Account length', 'mean')
    ).round(3)
    g2['Rata_churn'] = (g2['Rata_churn'] * 100).round(2).astype(str) + ' %'
    st.dataframe(g2, use_container_width=True)
    st.info("""
    **Interpretare:** Clienții cu plan de mesagerie vocală au o rată de churn mai mică, 
    indicând că acest serviciu este apreciat și contribuie la retenție. Promovarea acestui 
    plan ar putea reduce churn-ul general.
    """)

with tab3:
    g3 = df.groupby('Customer service calls').agg(
        Numar_clienti=('Churn', 'count'),
        Rata_churn=('Churn', 'mean')
    ).round(3)
    g3['Rata_churn_pct'] = (g3['Rata_churn'] * 100).round(2)
    st.dataframe(g3[['Numar_clienti', 'Rata_churn_pct']], use_container_width=True)
    st.warning("""
    **Interpretare critică:** Există o corelație puternică între numărul de apeluri la 
    serviciul clienți și rata de churn. Clienții care sună de 4+ ori au rate de churn 
    de peste 45%, comparativ cu sub 12% pentru cei cu 0-3 apeluri. 
    **Recomandare:** Echipa de Customer Service trebuie să rezolve problemele din prima 
    interacțiune și să implementeze un sistem de alertă la 3+ apeluri.
    """)

with tab4:
    g4 = df.groupby('State').agg(
        Numar_clienti=('Churn', 'count'),
        Rata_churn=('Churn', 'mean')
    ).round(3)
    g4['Rata_churn_pct'] = (g4['Rata_churn'] * 100).round(2)
    g4 = g4.sort_values('Rata_churn_pct', ascending=False).head(10)
    st.dataframe(g4[['Numar_clienti', 'Rata_churn_pct']], use_container_width=True)
    st.info("""
    **Interpretare:** Statele cu cele mai mari rate de churn ar trebui investigate separat — 
    posibile cauze includ concurența locală, calitatea acoperirii sau campanii agresive ale 
    competitorilor în zonele respective.
    """)

st.markdown("---")

st.header("Accesarea datelor cu `loc` și `iloc`")

st.markdown("""
**Definirea problemei:** Demonstrăm utilizarea celor două metode de indexare ale pandas:
- **`loc`** — indexare pe bază de **etichete** (nume coloane, valori index)
- **`iloc`** — indexare pe bază de **poziție numerică**

Aceste metode sunt esențiale pentru extragerea de subseturi specifice de date.
""")

col_l, col_i = st.columns(2)

with col_l:
    st.subheader("Folosind `loc` (etichete)")
    
    st.markdown("**Exemplu 1:** Clienți care au făcut churn ȘI au plan internațional")
    st.code("df.loc[(df['Churn']==True) & (df['International plan']=='Yes')]", language="python")
    rezultat_loc1 = df.loc[
        (df['Churn']==True) & (df['International plan']=='Yes'),
        ['State', 'Account length', 'Total day charge', 'Customer service calls']
    ].head(10)
    st.dataframe(rezultat_loc1, use_container_width=True)
    st.caption(f"Total clienți în acest segment: **{((df['Churn']==True) & (df['International plan']=='Yes')).sum()}**")
    
    st.markdown("**Exemplu 2:** Cost mediu zi pentru clienții cu vechime > 150 zile")
    cost_mediu_loc = df.loc[df['Account length'] > 150, 'Total day charge'].mean()
    st.metric("Cost mediu zi (vechime > 150)", f"${cost_mediu_loc:.2f}")

with col_i:
    st.subheader("Folosind `iloc` (poziție)")
    
    st.markdown("**Exemplu 1:** Primele 5 rânduri și primele 5 coloane")
    st.code("df.iloc[:5, :5]", language="python")
    st.dataframe(df.iloc[:5, :5], use_container_width=True)
    
    st.markdown("**Exemplu 2:** Ultimele 3 rânduri și ultimele 4 coloane")
    st.code("df.iloc[-3:, -4:]", language="python")
    st.dataframe(df.iloc[-3:, -4:], use_container_width=True)
    
    st.markdown("**Exemplu 3:** Rândurile pare, coloanele 5-10")
    st.code("df.iloc[::2, 5:10].head(5)", language="python")
    st.dataframe(df.iloc[::2, 5:10].head(5), use_container_width=True)

st.markdown("---")

st.header("Statistici descriptive")

col_st1, col_st2 = st.columns(2)

with col_st1:
    st.subheader("Variabile numerice")
    st.dataframe(df.describe().round(2), use_container_width=True)

with col_st2:
    st.subheader("Variabile categoriale")
    st.dataframe(df.describe(include=['object', 'bool']), use_container_width=True)

st.markdown("---")

st.subheader("Concluzii și recomandări de business")
st.success("""
**Insight-uri principale din EDA:**

1. **Rata generală de churn este ~14.5%**, peste media industriei, indicând nevoia de intervenție.
2. **Planul internațional crește riscul de churn de ~3x** — necesar review tarife și acoperire.
3. **Apelurile la Customer Service sunt un predictor puternic** — clienții care sună 4+ ori au probabilitate >40% să plece.
4. **Planul Voicemail reduce churn-ul** — clienții cu acest plan sunt mai loiali.

**Recomandări inițiale pentru extindere și retenție:**
- Implementarea unui sistem de **early warning** pentru clienții cu 3+ apeluri CS
- **Restructurarea planului internațional** cu pachete competitive
- **Promovarea Voicemail-ului** ca beneficiu adițional gratuit pentru clienții cu vechime > 100 zile
- **Campanii regionale targetate** în statele cu rate ridicate de churn
""")
