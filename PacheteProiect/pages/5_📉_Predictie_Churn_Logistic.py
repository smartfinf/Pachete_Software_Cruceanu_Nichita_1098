import streamlit as st
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

st.set_page_config(page_title="Predicție Churn", layout="wide")

st.title("Predicția churn-ului cu Regresie Logistică")
st.markdown("---")

st.markdown(
    """
**Definirea problemei:** Orange Telecom vrea să identifice din timp clienții cu risc ridicat de churn.
Un model de clasificare poate estima probabilitatea ca un client să renunțe, astfel încât echipa de
retenție să intervină proactiv.

**Metodă:** **Regresie Logistică** (scikit-learn). Modelul estimează:

`P(Churn=1 | X) = 1 / (1 + e^-(β0 + β1 x1 + ... + βp xp))`

**Rezultate urmărite:** metrici (accuracy, precision, recall, F1, AUC) + matrice de confuzie + ROC.
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

candidate_features = [
    'Account length',
    'International plan',
    'Voice mail plan',
    'Number vmail messages',
    'Total day charge',
    'Total eve charge',
    'Total night charge',
    'Total intl charge',
    'Customer service calls',
    'State_freq',
    'Total charge',
]

available_features = [c for c in candidate_features if c in df_encoded.columns]

st.header("1️⃣ Setarea variabilelor și împărțirea train/test")

features_selected = st.multiselect(
    "Alege variabilele explicative (X)",
    options=available_features,
    default=available_features,
)

col_a, col_b, col_c = st.columns(3)
with col_a:
    test_size = st.slider("Proporție test", 0.1, 0.4, 0.2, 0.05)
with col_b:
    random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)
with col_c:
    class_weight_opt = st.selectbox("Class weight", options=["None", "balanced"], index=1)

if len(features_selected) == 0:
    st.warning("Selectează cel puțin o variabilă explicativă.")
    st.stop()

X = df_encoded[features_selected].copy()
y = df_encoded['Churn'].astype(int).copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=float(test_size),
    random_state=int(random_state),
    stratify=y,
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class_weight = None if class_weight_opt == "None" else "balanced"

st.caption(
    f"Train: {X_train.shape[0]:,} rânduri | Test: {X_test.shape[0]:,} rânduri | "
    f"Churn train: {y_train.mean()*100:.2f}%"
)

st.markdown("---")


st.header("2️⃣ Antrenare model Logistic Regression")

col_p1, col_p2, col_p3 = st.columns(3)
with col_p1:
    C = st.select_slider("Regularizare inversă (C)", options=[0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0], value=1.0)
with col_p2:
    max_iter = st.slider("Max iter", 100, 2000, 500, 100)
with col_p3:
    threshold = st.slider("Prag decizie (threshold)", 0.1, 0.9, 0.5, 0.05)

model = LogisticRegression(
    C=float(C),
    max_iter=int(max_iter),
    solver="lbfgs",
    class_weight=class_weight,
)
model.fit(X_train_scaled, y_train)

proba_test = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (proba_test >= float(threshold)).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, proba_test)

st.subheader("Metrici model")
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
col_m1.metric("Accuracy", f"{acc:.3f}")
col_m2.metric("Precision", f"{prec:.3f}")
col_m3.metric("Recall", f"{rec:.3f}")
col_m4.metric("F1", f"{f1:.3f}")
col_m5.metric("AUC", f"{auc:.3f}")

st.markdown("---")


st.header("3️⃣ Matrice confuzie și ROC")

cm = confusion_matrix(y_test, y_pred)
col_cm, col_roc = st.columns(2)

with col_cm:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel('Predicție')
    ax.set_ylabel('Adevăr')
    ax.set_title('Matrice de confuzie')
    ax.set_xticklabels(['Non-churn', 'Churn'])
    ax.set_yticklabels(['Non-churn', 'Churn'])
    plt.tight_layout()
    st.pyplot(fig)

with col_roc:
    fpr, tpr, _ = roc_curve(y_test, proba_test)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='#FF6F00', linewidth=2, label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")


st.header("4️⃣ Interpretare: ce variabile cresc riscul de churn")

coef_df = pd.DataFrame({
    'Variabilă': features_selected,
    'Coeficient (β)': model.coef_[0],
})
coef_df['|Coef|'] = coef_df['Coeficient (β)'].abs()
coef_df = coef_df.sort_values('|Coef|', ascending=False).drop(columns=['|Coef|'])

st.dataframe(coef_df, use_container_width=True, hide_index=True)

st.info(
    """
**Interpretare:**
- Un coeficient **pozitiv** înseamnă că, pe măsură ce variabila crește, crește și log-șansa de churn.
- Un coeficient **negativ** înseamnă că variabila este asociată cu retenție (scade riscul).

În telecom, un semnal tipic este `Customer service calls` (nemulțumire / probleme repetate).
"""
)

st.markdown("---")


st.subheader("Interpretare economică și acțiuni")
st.success(
    """
**Concluzie:** Modelul logistic oferă o estimare a riscului de churn per client.

**Recomandări de business (retenție):**
1. Targetare proactivă pentru clienții cu probabilitate de churn ridicată (de ex. > 0.6).
2. Workflow automat: dacă un client depășește 3 apeluri la Customer Service → ofertă personalizată / escaladare.
3. Optimizare plan internațional pentru reducerea churn-ului (analize pe sub-segmente).
"""
)
