import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Vizualizări Interactive", layout="wide")

st.title("Vizualizări Interactive cu matplotlib, seaborn și plotly")
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

st.markdown("""
**Definirea problemei:** Realizăm un panou complet de vizualizări pentru a înțelege 
distribuția datelor, corelațiile între variabile și diferențele între segmentele 
de clienți (churn vs non-churn). Utilizăm 3 biblioteci diferite pentru a demonstra 
versatilitatea în vizualizare.
""")

st.header("1. Distribuția variabilelor cheie (matplotlib)")

col_var = st.selectbox(
    "Selectează variabila pentru analiză:",
    ['Account length', 'Total day minutes', 'Total day charge',
     'Total eve minutes', 'Total night minutes', 'Customer service calls',
     'Number vmail messages']
)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].hist(df[col_var], bins=40, color='#FF6F00', edgecolor='black', alpha=0.8)
axes[0].axvline(df[col_var].mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {df[col_var].mean():.2f}')
axes[0].axvline(df[col_var].median(), color='blue', linestyle='--', linewidth=2, label=f'Mediana: {df[col_var].median():.2f}')
axes[0].set_xlabel(col_var)
axes[0].set_ylabel('Frecvență')
axes[0].set_title(f'Distribuția {col_var}')
axes[0].legend()
axes[0].grid(alpha=0.3)

df_no_churn = df[df['Churn'] == False][col_var]
df_churn = df[df['Churn'] == True][col_var]
axes[1].hist(df_no_churn, bins=30, color='green', alpha=0.6, label='Non-churn', edgecolor='black')
axes[1].hist(df_churn, bins=30, color='red', alpha=0.6, label='Churn', edgecolor='black')
axes[1].set_xlabel(col_var)
axes[1].set_ylabel('Frecvență')
axes[1].set_title(f'{col_var}: Churn vs Non-Churn')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

st.header("2. Analiză corelații și distribuții (seaborn)")

col_s1, col_s2 = st.columns(2)

with col_s1:
    st.subheader("Heatmap corelații")
    numeric_cols_for_corr = [
        'Account length', 'Number vmail messages', 'Total day minutes',
        'Total day charge', 'Total eve charge', 'Total night charge',
        'Total intl charge', 'Customer service calls'
    ]
    
    corr = df[numeric_cols_for_corr + ['Churn']].copy()
    corr['Churn'] = corr['Churn'].astype(int)
    corr_matrix = corr.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn_r', center=0, 
                fmt='.2f', square=True, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Matrice de corelații (Pearson)', fontsize=14, pad=10)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.caption("""
    **Interpretare:** Coloanele `Total *minutes` și `Total *charge` sunt aproape perfect 
    corelate (r ≈ 1.00) deoarece costul = minute × tarif fix. Numărul de apeluri la 
    serviciul clienți are cea mai puternică corelație pozitivă cu churn.
    """)

with col_s2:
    st.subheader("Boxplot — variabila vs Churn")
    
    boxplot_var = st.selectbox(
        "Variabila de comparat",
        ['Total day minutes', 'Total day charge', 'Total eve charge',
         'Customer service calls', 'Account length', 'Total intl charge'],
        key='boxplot_var'
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(data=df, x='Churn', y=boxplot_var, palette=['#28a745', '#dc3545'], ax=ax)
    sns.stripplot(data=df, x='Churn', y=boxplot_var, color='black', 
                  alpha=0.2, size=2, ax=ax)
    ax.set_title(f'{boxplot_var} pentru clienți Churn vs Non-Churn', fontsize=13)
    ax.set_xlabel('A renunțat la servicii (Churn)')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")

st.header("3. Grafice interactive (plotly)")

st.subheader("Scatter 3D: Vechime × Cost zi × Apeluri CS, colorate după Churn")

fig_3d = px.scatter_3d(
    df.sample(n=min(1500, len(df)), random_state=42),
    x='Account length',
    y='Total day charge',
    z='Customer service calls',
    color='Churn',
    color_discrete_map={True: '#dc3545', False: '#28a745'},
    title="Distribuția clienților în spațiul 3D",
    labels={'Churn': 'Churn'},
    opacity=0.7,
    size_max=10
)
fig_3d.update_layout(height=600)
st.plotly_chart(fig_3d, use_container_width=True)

st.markdown("---")

col_p1, col_p2 = st.columns(2)

with col_p1:
    st.subheader("Sunburst: distribuția pe planuri")
    df_sun = df.copy()
    df_sun['Churn_label'] = df_sun['Churn'].map({True: 'Churn', False: 'Loial'})
    fig_sun = px.sunburst(
        df_sun,
        path=['International plan', 'Voice mail plan', 'Churn_label'],
        title="Distribuția clienților: Plan Internațional → Voicemail → Churn",
        color='Churn_label',
        color_discrete_map={'Churn': '#dc3545', 'Loial': '#28a745'}
    )
    fig_sun.update_layout(height=500)
    st.plotly_chart(fig_sun, use_container_width=True)

with col_p2:
    st.subheader("Bar chart: Churn rate pe Apeluri CS")
    
    cs_data = df.groupby('Customer service calls').agg(
        clienti=('Churn', 'count'),
        churn_rate=('Churn', 'mean')
    ).reset_index()
    cs_data['churn_rate'] = cs_data['churn_rate'] * 100
    
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=cs_data['Customer service calls'],
        y=cs_data['churn_rate'],
        text=[f'{v:.1f}%' for v in cs_data['churn_rate']],
        textposition='outside',
        marker_color=['#28a745' if v < 25 else '#ffc107' if v < 50 else '#dc3545' 
                      for v in cs_data['churn_rate']],
        name='Rata Churn (%)'
    ))
    fig_bar.update_layout(
        title="Rata de churn vs număr apeluri la Serviciul Clienți",
        xaxis_title="Număr apeluri Serviciu Clienți",
        yaxis_title="Rata churn (%)",
        height=500,
        showlegend=False
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

st.subheader("Hartă SUA: Rata de churn pe stat (choropleth)")

state_churn = df.groupby('State').agg(
    clienti=('Churn', 'count'),
    rata_churn=('Churn', 'mean')
).reset_index()
state_churn['rata_churn_pct'] = (state_churn['rata_churn'] * 100).round(2)

fig_map = px.choropleth(
    state_churn,
    locations='State',
    locationmode='USA-states',
    color='rata_churn_pct',
    scope='usa',
    color_continuous_scale='Reds',
    title='Rata de churn pe stat (%)',
    labels={'rata_churn_pct': 'Rata Churn (%)'},
    hover_data=['clienti']
)
fig_map.update_layout(height=600)
st.plotly_chart(fig_map, use_container_width=True)

st.markdown("---")

st.subheader("Interpretare economică a vizualizărilor")
st.success("""
**Observații principale:**

1. **Distribuții comparative (matplotlib):** Clienții care fac churn au consum semnificativ 
   mai ridicat în timpul zilei și mai multe apeluri la serviciul clienți.

2. **Matricea de corelații (seaborn):** Există multicoliniaritate perfectă între `*minutes` 
   și `*charge` (același semnal economic). Pentru modelele de regresie, vom păstra doar 
   variabilele `charge` pentru a evita probleme de multicoliniaritate.

3. **Scatter 3D (plotly):** Clienții cu vechime mare, cost ridicat ziua și 4+ apeluri CS 
   formează un cluster distinct cu risc maxim de churn.

4. **Harta SUA:** Există variabilitate geografică importantă — câteva state au rate de 
   churn de 2-3x peste media națională. Aceste piețe necesită strategii de retenție 
   personalizate.

**Recomandări:**
- **Bugetul de retenție** trebuie alocat prioritar pentru clienții cu cost zi > $40 și 
  apeluri CS > 3 (segmentul cu cel mai mare risc).
- **Investigare aprofundată** a statelor cu rate >25% — posibilă oportunitate de extindere 
  prin oferte locale competitive.
""")
