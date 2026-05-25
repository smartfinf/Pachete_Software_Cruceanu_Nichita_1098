import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path

st.set_page_config(page_title="Importul Datelor", layout="wide")

st.title("Importul Datelor și Pregătirea pentru Analiză")
st.markdown("---")

st.header("1. Importul fișierului CSV în pandas")

st.markdown("""
**Definirea problemei:** Citim setul de date Orange Telecom Customer Churn care conține 
informații despre 3.333 de clienți (durata abonamentului, planuri active, consum de 
minute pe diverse intervale orare, apeluri la serviciul clienți și statusul de churn).
""")

@st.cache_data
def load_data_default():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    combined_path = data_dir / "orange_telecom_churn.csv"
    if combined_path.exists():
        return pd.read_csv(combined_path)

    part_80 = project_root / "churn-bigml-80.csv"
    part_20 = project_root / "churn-bigml-20.csv"
    if part_80.exists() and part_20.exists():
        df_80 = pd.read_csv(part_80)
        df_20 = pd.read_csv(part_20)
        return pd.concat([df_80, df_20], ignore_index=True)

    return None

df = load_data_default()
if df is None:
    st.error("Nu am găsit dataset-ul local. Pune fișierele `churn-bigml-80.csv` și `churn-bigml-20.csv` în folderul proiectului.")
    st.stop()

if 'Churn' in df.columns and df['Churn'].dtype == object:
    churn_map = {'true': True, 'false': False}
    df['Churn'] = df['Churn'].astype(str).str.strip().str.lower().map(churn_map)

numeric_cols_coerce = [
    'Account length', 'Area code', 'Number vmail messages',
    'Total day minutes', 'Total day calls', 'Total day charge',
    'Total eve minutes', 'Total eve calls', 'Total eve charge',
    'Total night minutes', 'Total night calls', 'Total night charge',
    'Total intl minutes', 'Total intl calls', 'Total intl charge',
    'Customer service calls'
]
for c in numeric_cols_coerce:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

col1, col2, col3, col4 = st.columns(4)
col1.metric("Înregistrări", f"{df.shape[0]:,}")
col2.metric("Variabile", df.shape[1])
col3.metric("Rata churn", f"{df['Churn'].mean()*100:.2f}%")
col4.metric("State (SUA)", df['State'].nunique())

st.subheader("Primele 10 rânduri")
st.dataframe(df.head(10), use_container_width=True)

with st.expander("Descrierea variabilelor"):
    descriere = pd.DataFrame({
        "Variabilă": df.columns.tolist(),
        "Tip": [str(df[c].dtype) for c in df.columns],
        "Descriere": [
            "Statul din SUA în care locuiește clientul (50 state + DC)",
            "Durata contului în zile (vechimea ca abonat)",
            "Codul ariei telefonice (408, 415, 510)",
            "Are plan internațional? (Yes/No)",
            "Are plan de mesagerie vocală? (Yes/No)",
            "Număr mesaje voicemail",
            "Total minute conversație în timpul zilei",
            "Număr total apeluri în timpul zilei",
            "Cost total apeluri în timpul zilei ($)",
            "Total minute conversație seara",
            "Număr total apeluri seara",
            "Cost total apeluri seara ($)",
            "Total minute conversație noaptea",
            "Număr total apeluri noaptea",
            "Cost total apeluri noaptea ($)",
            "Total minute apeluri internaționale",
            "Număr total apeluri internaționale",
            "Cost total apeluri internaționale ($)",
            "Număr apeluri către serviciul clienți",
            "Variabilă țintă: A renunțat clientul la servicii? (True/False)"
        ]
    })
    st.dataframe(descriere, use_container_width=True, hide_index=True)

st.markdown("---")

st.header("2. Tratarea valorilor lipsă")

st.markdown("""
**Definirea problemei:** În setul de date există valori lipsă pe câteva coloane numerice. 
Pentru a putea aplica modele de machine learning trebuie să imputăm aceste valori. 
Vom folosi **imputarea cu mediana** pentru variabilele numerice (rezistentă la valorile aberante).

**Metodă:** Pentru fiecare coloană numerică cu valori lipsă, înlocuim NaN cu mediana coloanei.
""")

missing_before = df.isnull().sum()
missing_before = missing_before[missing_before > 0]

if len(missing_before) > 0:
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Înainte de tratare")
        df_missing = pd.DataFrame({
            "Variabilă": missing_before.index,
            "Valori lipsă": missing_before.values,
            "Procent (%)": (missing_before.values / len(df) * 100).round(2)
        })
        st.dataframe(df_missing, hide_index=True, use_container_width=True)
    
    df_clean = df.copy()
    for col in missing_before.index:
        median_val = df_clean[col].median()
        df_clean[col].fillna(median_val, inplace=True)
        st.caption(f"Coloana **{col}**: înlocuit cu mediana = `{median_val:.2f}`")
    
    with col_b:
        st.subheader("După tratare")
        st.success(f"Total valori lipsă: **{df_clean.isnull().sum().sum()}**")
        st.dataframe(
            pd.DataFrame({
                "Variabilă": missing_before.index,
                "Valori lipsă": [df_clean[c].isnull().sum() for c in missing_before.index]
            }),
            hide_index=True,
            use_container_width=True
        )
else:
    st.info("Nu există valori lipsă în setul de date.")
    df_clean = df.copy()

st.session_state['df_clean'] = df_clean

st.markdown("---")

st.header("3. Codificarea variabilelor categoriale și scalarea")

st.markdown("""
**Definirea problemei:** Algoritmii de machine learning lucrează doar cu valori numerice. 
Trebuie să convertim variabilele categoriale (`International plan`, `Voice mail plan`, `State`) 
în numere. De asemenea, pentru clusterizare și regresie logistică, e necesară 
**standardizarea** variabilelor numerice (medie 0, deviație standard 1) pentru ca toate 
să contribuie egal la calculul distanțelor.

**Metode:**
- **Label Encoding** pentru variabile binare (`Yes/No` → `1/0`)
- **Frequency Encoding** pentru `State` (folosim frecvența ca proxy pentru piață)
- **StandardScaler** pentru variabilele numerice
""")

df_encoded = df_clean.copy()

df_encoded['International plan'] = df_encoded['International plan'].map({'Yes': 1, 'No': 0})
df_encoded['Voice mail plan'] = df_encoded['Voice mail plan'].map({'Yes': 1, 'No': 0})
df_encoded['Churn'] = df_encoded['Churn'].astype(int)

state_freq = df_encoded['State'].value_counts(normalize=True).to_dict()
df_encoded['State_freq'] = df_encoded['State'].map(state_freq)

col_e1, col_e2 = st.columns(2)

with col_e1:
    st.subheader("Înainte de codificare")
    st.dataframe(
        df_clean[['International plan', 'Voice mail plan', 'State', 'Churn']].head(10),
        use_container_width=True
    )

with col_e2:
    st.subheader("După codificare")
    st.dataframe(
        df_encoded[['International plan', 'Voice mail plan', 'State_freq', 'Churn']].head(10),
        use_container_width=True
    )

st.subheader("Standardizarea variabilelor numerice (StandardScaler)")

numeric_cols = [
    'Account length', 'Number vmail messages',
    'Total day minutes', 'Total day calls', 'Total day charge',
    'Total eve minutes', 'Total eve calls', 'Total eve charge',
    'Total night minutes', 'Total night calls', 'Total night charge',
    'Total intl minutes', 'Total intl calls', 'Total intl charge',
    'Customer service calls'
]

scaler = StandardScaler()
df_scaled = df_encoded.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

col_s1, col_s2 = st.columns(2)

with col_s1:
    st.caption("Statistici înainte de scalare")
    st.dataframe(
        df_encoded[numeric_cols[:5]].describe().round(2),
        use_container_width=True
    )

with col_s2:
    st.caption("Statistici după scalare (medie≈0, std≈1)")
    st.dataframe(
        df_scaled[numeric_cols[:5]].describe().round(2),
        use_container_width=True
    )

st.session_state['df_encoded'] = df_encoded
st.session_state['df_scaled'] = df_scaled
st.session_state['numeric_cols'] = numeric_cols

st.markdown("---")

st.subheader("Interpretare economică")
st.success("""
**Concluzii preliminare:**

- Setul de date conține **3.333 clienți** Orange Telecom, dintre care **14.5% au renunțat la servicii** (churn=True).
  Această rată este peste media industriei telecom (10-12%), indicând o problemă reală de retenție.
- Variabilele categoriale au fost codificate corespunzător, păstrând semnificația economică.
- Datele standardizate sunt pregătite pentru clusterizare și regresie logistică (paginile 4 și 5).
- **Recomandare pentru business:** Înainte de modelare, e important să identificăm caracteristicile 
  segmentului care a renunțat — aceasta va fi analiza din paginile următoare.
""")
