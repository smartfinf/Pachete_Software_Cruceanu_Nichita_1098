import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from pathlib import Path

st.set_page_config(page_title="Clusterizare Clienți", layout="wide")

st.title("Segmentarea Clienților cu K-Means")
st.markdown("---")

st.markdown("""
**Definirea problemei:** Vrem să identificăm segmente naturale de clienți Orange Telecom 
pe baza comportamentului lor de consum. Acest lucru permite echipei de marketing să 
construiască campanii diferențiate (cross-selling, retenție, upgrade) pentru fiecare segment.

**Metodă:** Algoritmul **K-Means** din scikit-learn — partiționează clienții în K grupuri 
astfel încât suma distanțelor pătratice în interiorul fiecărui grup să fie minimă.

**Variabile folosite pentru clusterizare:** consum minute (zi/seară/noapte/internațional), 
cost total, durată cont, apeluri la serviciul clienți.
""")

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

features_for_cluster = [
    'Account length', 'Total day minutes', 'Total eve minutes',
    'Total night minutes', 'Total intl minutes', 'Total day charge',
    'Customer service calls', 'Number vmail messages'
]

X = df[features_for_cluster].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.header("1. Determinarea numărului optim de clustere")

st.markdown("""
Folosim două metode complementare:
- **Metoda Cotului (Elbow Method):** căutăm punctul în care reducerea inerției încetinește
- **Coeficientul Silhouette:** valori mai mari indică clustere mai bine definite (max = 1)
""")

@st.cache_data
def compute_elbow_silhouette(X_scaled_array):
    inertias = []
    silhouettes = []
    k_range = range(2, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled_array)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled_array, labels))
    return list(k_range), inertias, silhouettes

k_range, inertias, silhouettes = compute_elbow_silhouette(X_scaled)

col_e, col_s = st.columns(2)

with col_e:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_range, inertias, 'o-', color='#FF6F00', linewidth=2, markersize=8)
    ax.set_xlabel('Număr clustere (K)')
    ax.set_ylabel('Inerția (suma distanțelor pătratice)')
    ax.set_title('Metoda Cotului (Elbow Method)')
    ax.grid(alpha=0.3)
    ax.axvline(x=4, color='red', linestyle='--', alpha=0.5, label='K=4 ales')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

with col_s:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_range, silhouettes, 'o-', color='#1f77b4', linewidth=2, markersize=8)
    ax.set_xlabel('Număr clustere (K)')
    ax.set_ylabel('Coeficient Silhouette')
    ax.set_title('Analiza Silhouette')
    ax.grid(alpha=0.3)
    ax.axvline(x=4, color='red', linestyle='--', alpha=0.5, label='K=4 ales')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

metrici_df = pd.DataFrame({
    'K': k_range,
    'Inertie': [round(i, 2) for i in inertias],
    'Silhouette': [round(s, 4) for s in silhouettes]
})
st.dataframe(metrici_df, use_container_width=True, hide_index=True)

st.markdown("---")

st.header("2. Aplicarea K-Means cu K ales")

k_choice = st.slider("Selectează numărul de clustere (K):", 2, 8, 4)

kmeans = KMeans(n_clusters=k_choice, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

silhouette_final = silhouette_score(X_scaled, df['Cluster'])
col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Număr clustere", k_choice)
col_m2.metric("Silhouette Score", f"{silhouette_final:.4f}")
col_m3.metric("Inerție", f"{kmeans.inertia_:.2f}")

st.subheader("Vizualizarea clusterelor în spațiul PCA 2D")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['Cluster'] = df['Cluster'].astype(str)
df_pca['Churn'] = df['Churn']

fig_pca = px.scatter(
    df_pca, x='PC1', y='PC2', color='Cluster',
    symbol='Churn',
    title=f'Clusterele K-Means proiectate în spațiul PCA (varianță explicată: {pca.explained_variance_ratio_.sum()*100:.1f}%)',
    color_discrete_sequence=px.colors.qualitative.Set1,
    opacity=0.7
)
fig_pca.update_layout(height=600)
st.plotly_chart(fig_pca, use_container_width=True)

st.markdown("---")

st.header("3. Profilul fiecărui cluster (caracteristici medii)")

cluster_profile = df.groupby('Cluster').agg({
    'Account length': 'mean',
    'Total day minutes': 'mean',
    'Total day charge': 'mean',
    'Total eve charge': 'mean',
    'Total night charge': 'mean',
    'Total intl charge': 'mean',
    'Customer service calls': 'mean',
    'Number vmail messages': 'mean',
    'Churn': 'mean'
}).round(2)

cluster_profile['Numar clienti'] = df.groupby('Cluster').size()
cluster_profile['Rata Churn %'] = (cluster_profile['Churn'] * 100).round(2)
cluster_profile = cluster_profile.drop('Churn', axis=1)

cols_order = ['Numar clienti', 'Rata Churn %', 'Account length', 'Total day minutes',
              'Total day charge', 'Total eve charge', 'Total night charge', 
              'Total intl charge', 'Customer service calls', 'Number vmail messages']
cluster_profile = cluster_profile[cols_order]

st.dataframe(cluster_profile.style.background_gradient(subset=['Rata Churn %'], cmap='Reds'),
             use_container_width=True)

st.subheader("Rata de churn pe fiecare cluster")

fig_churn_cluster = px.bar(
    cluster_profile.reset_index(),
    x='Cluster',
    y='Rata Churn %',
    text='Rata Churn %',
    title=f'Rata de churn pe fiecare cluster (K={k_choice})',
    color='Rata Churn %',
    color_continuous_scale='Reds'
)
fig_churn_cluster.update_traces(textposition='outside')
fig_churn_cluster.update_layout(height=500)
st.plotly_chart(fig_churn_cluster, use_container_width=True)

st.markdown("---")

st.header("Interpretare economică și strategii pentru fiecare segment")

if k_choice == 4:
    st.success("""
    **Pentru K=4 clustere identificate, interpretarea tipică este:**
    
    **Cluster A — Clienți Premium loiali**  
    Consum ridicat pe toate intervalele, vechime mare, puține apeluri CS, rată churn mică (~5-10%).  
    **Strategie:** Programe de fidelizare premium, upgrade-uri (5G, conținut exclusiv), referral.
    
    **Cluster B — Clienți Standard moderați**  
    Consum mediu, profilul tipic al unei familii.  
    **Strategie:** Cross-selling (bundles TV+Internet), abonament familial cu discount.
    
    **Cluster C — Clienți Light-users**  
    Consum redus, posibil pensionari sau utilizatori secundari.  
    **Strategie:** Migrare la planuri prepay sau abonamente light cu costuri mai mici (anti-churn).
    
    **Cluster D — Clienți cu risc maxim de churn**  
    Apeluri multiple la CS, plan internațional, cost ridicat — nemulțumiți.  
    **Strategie:** Intervenție imediată — retention specialist, discount loial, escaladare.
    """)
else:
    st.info(f"""
    **K={k_choice} clustere identificate.** Inspectează tabelul de mai sus pentru a identifica:
    - Clusterele cu rată de churn ridicată (>20%) → necesită intervenție de retenție.
    - Clusterele cu rată de churn scăzută (<10%) → candidați pentru programe de fidelizare și referral.
    - Clusterele cu consum mare → oportunități de upselling.
    - Clusterele cu multe apeluri CS → necesită îmbunătățiri ale calității serviciului.
    """)

st.markdown("---")
st.subheader("Recomandări pentru extinderea afacerii")
st.warning("""
**Acțiuni prioritare bazate pe clusterizare:**

1. **Buget de retenție diferențiat** — concentrarea efortului pe clusterele cu rată ridicată de churn
2. **Pachete personalizate** pentru fiecare segment (Premium, Standard, Light, At-Risk)
3. **Programe de upgrade** pentru clusterele cu potențial de creștere (Light → Standard)
4. **Investigare profundă a clusterului cu risc maxim** — interviuri, analiză cauzală pentru a 
   identifica problemele specifice și a rezolva root cause-ul
5. **Modelul poate fi rulat lunar** pentru a monitoriza migrarea clienților între clustere și a 
   detecta riscurile timpuriu
""")
