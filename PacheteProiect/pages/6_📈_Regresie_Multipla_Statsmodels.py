import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Regresie Multiplă", layout="wide")

st.title("Regresie multiplă (OLS) pentru factura clientului")
st.markdown("---")

st.markdown(
    """
**Definirea problemei:** Orange Telecom vrea să înțeleagă **ce factori determină factura totală**
(totalul costurilor zilnice + seară + noapte + internațional). Rezultatul ajută la:

- identificarea driverilor de venit
- personalizarea planurilor (upsell)
- calibrarea discount-urilor fără a afecta profitabilitatea

**Metodă:** **Regresie multiplă OLS** (statsmodels), care estimează o relație liniară:

`Y = β0 + β1 x1 + ... + βp xp + ε`

**Metrici:** R², RMSE, MAE + semnificația coeficienților (p-values).
"""
)


def _load_dataset_fallback() -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[1]
    combined_path = project_root / "data" / "orange_telecom_churn.csv"
    if combined_path.exists():
        df_local = pd.read_csv(combined_path)
    else:
        part_80 = project_root / "churn-bigml-80.csv"
        part_20 = project_root / "churn-bigml-20.csv"
        if part_80.exists() and part_20.exists():
            df_local = pd.concat([pd.read_csv(part_80), pd.read_csv(part_20)], ignore_index=True)
        else:
            return None

    if 'Churn' in df_local.columns and df_local['Churn'].dtype == object:
        churn_map = {'true': True, 'false': False}
        df_local['Churn'] = df_local['Churn'].astype(str).str.strip().str.lower().map(churn_map)

    return df_local


if 'df_encoded' in st.session_state:
    df_encoded = st.session_state['df_encoded'].copy()
else:
    df_raw = _load_dataset_fallback()
    if df_raw is None:
        st.error("Nu am găsit dataset-ul local. Pune `churn-bigml-80.csv` și `churn-bigml-20.csv` în folderul proiectului.")
        st.stop()

    df_encoded = df_raw.copy()
    df_encoded['International plan'] = df_encoded['International plan'].map({'Yes': 1, 'No': 0})
    df_encoded['Voice mail plan'] = df_encoded['Voice mail plan'].map({'Yes': 1, 'No': 0})
    df_encoded['Churn'] = df_encoded['Churn'].astype(int)
    state_freq = df_encoded['State'].value_counts(normalize=True).to_dict()
    df_encoded['State_freq'] = df_encoded['State'].map(state_freq)

df_encoded['Total charge'] = (
    df_encoded['Total day charge']
    + df_encoded['Total eve charge']
    + df_encoded['Total night charge']
    + df_encoded['Total intl charge']
)

st.header("1. Definirea variabilelor")

st.caption("Variabila dependentă (Y): `Total charge` (zi + seară + noapte + internațional)")

candidate_features = [
    'Account length',
    'International plan',
    'Voice mail plan',
    'Number vmail messages',
    'Total day minutes',
    'Total eve minutes',
    'Total night minutes',
    'Total intl minutes',
    'Customer service calls',
    'State_freq',
]

available_features = [c for c in candidate_features if c in df_encoded.columns]

features_selected = st.multiselect(
    "Alege variabilele explicative (X)",
    options=available_features,
    default=available_features,
)

col_a, col_b = st.columns(2)
with col_a:
    test_size = st.slider("Proporție test", 0.1, 0.4, 0.2, 0.05)
with col_b:
    random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)

if len(features_selected) == 0:
    st.warning("Selectează cel puțin o variabilă explicativă.")
    st.stop()

X = df_encoded[features_selected].copy()
y = df_encoded['Total charge'].astype(float).copy()

X = X.replace([np.inf, -np.inf], np.nan)
y = y.replace([np.inf, -np.inf], np.nan)

data_model = pd.concat([X, y.rename('Total charge')], axis=1).dropna()
X = data_model[features_selected]
y = data_model['Total charge']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=float(test_size),
    random_state=int(random_state),
)

st.caption(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

st.markdown("---")


st.header("2. Estimarea modelului OLS (statsmodels)")

X_train_sm = sm.add_constant(X_train, has_constant='add')
X_test_sm = sm.add_constant(X_test, has_constant='add')

model = sm.OLS(y_train, X_train_sm)
results = model.fit()

st.subheader("Rezumat model")
with st.expander("Afișează summary()"):
    st.text(results.summary())

st.markdown("---")


st.header("3. Predicții și metrici")

y_pred = results.predict(X_test_sm)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("R² (test)", f"{r2:.3f}")
col_m2.metric("RMSE (test)", f"{rmse:.3f}")
col_m3.metric("MAE (test)", f"{mae:.3f}")
col_m4.metric("R² (train)", f"{results.rsquared:.3f}")

st.markdown("---")


st.header("4. Interpretarea coeficienților")

coef_table = pd.DataFrame({
    'Coeficient (β)': results.params,
    'P-value': results.pvalues,
}).reset_index().rename(columns={'index': 'Variabilă'})

coef_table['Semnificativ (p<0.05)'] = coef_table['P-value'] < 0.05
coef_table = coef_table.sort_values('Coeficient (β)', ascending=False)

st.dataframe(coef_table, use_container_width=True, hide_index=True)

st.markdown("---")


st.header("5. Diagnostic: Predicții vs Real")

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, y_pred, alpha=0.4, color='#FF6F00')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='black', linewidth=1)
ax.set_xlabel('Total charge real')
ax.set_ylabel('Total charge prezis')
ax.set_title('Real vs Prezis (set test)')
ax.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig)

residuals = y_test - y_pred
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(residuals, bins=30, kde=True, ax=ax, color='#1f77b4')
ax.set_title('Distribuția reziduurilor')
ax.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")


st.subheader("Interpretare economică")
st.success(
    """
**Cum folosești rezultatul în business:**

- Variabilele cu coeficient mare și semnificativ statistic (p<0.05) sunt **driveri ai facturii**.
- Dacă `International plan` este semnificativ pozitiv, planul internațional aduce venit, dar (din paginile anterioare)
  poate crește churn-ul → e nevoie de echilibru între ARPU și retenție.
- Dacă minutele (zi/seară/noapte) sunt dominante, se pot crea pachete targetate pe intervalul de consum.

**Recomandare:** folosește modelul împreună cu segmentarea și predicția churn:
- upsell pe segmente loiale
- discount/retention pe segmente cu risc mare de churn
"""
)
