# Streamlit App for lncRNA-Drug Resistance Analysis

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
import shap
from io import StringIO
from gseapy import enrichr

st.set_page_config(page_title="lncRNA Drug Resistance Analysis", layout="wide")

st.title("🧬 lncRNA-Drug Resistance Predictor")
st.markdown("Upload your lncRNA expression matrix, drug labels, and survival data.")

# File uploads
expr_file = st.file_uploader("Upload lncRNA Expression CSV (samples x lncRNAs):", type="csv")
label_file = st.file_uploader("Upload Drug Resistance Labels CSV:", type="csv")
survival_file = st.file_uploader("Upload Survival Metadata CSV:", type="csv")

if expr_file and label_file:
    expr = pd.read_csv(expr_file, index_col=0)
    labels = pd.read_csv(label_file, index_col=0)
    common = expr.index.intersection(labels.index)
    X = expr.loc[common]
    y = labels.loc[common, 'response']

    # Filter low-expressed lncRNAs
    X = X.loc[:, (X.mean() > 0.3)]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    y_prob = rf.predict_proba(X_scaled)[:, 1]
    auc = roc_auc_score(y, y_prob)

    # Top features
    importances = rf.feature_importances_
    feat_df = pd.DataFrame({'lncRNA': X.columns, 'importance': importances})
    top_feats = feat_df.sort_values(by='importance', ascending=False).head(20)

    st.success(f"Random Forest AUC: {auc:.3f}")
    st.subheader("🔍 Top 20 Important lncRNAs")
    st.dataframe(top_feats.reset_index(drop=True))

    # Heatmap
    st.subheader("📊 Heatmap of Top lncRNAs")
    top_expr = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)[top_feats['lncRNA']]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(top_expr.T, cmap="vlag", cbar=True, ax=ax)
    st.pyplot(fig)

    # Volcano Plot
    if y.nunique() == 2:
        st.subheader("🌋 Volcano Plot")
        from scipy.stats import ttest_ind
        group1 = X[y == 0]
        group2 = X[y == 1]
        t_stat, p_val = ttest_ind(group2, group1)
        log_fc = np.log2(group2.mean() / group1.mean())
        volcano = pd.DataFrame({"log2FC": log_fc, "-log10(p)": -np.log10(p_val)}, index=X.columns)
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=volcano, x="log2FC", y="-log10(p)", ax=ax2)
        ax2.axhline(1.3, ls='--', c='red')
        ax2.axvline(1, ls='--', c='blue')
        ax2.axvline(-1, ls='--', c='blue')
        st.pyplot(fig2)

    # SHAP Analysis
    st.subheader("🧠 SHAP Interpretability")
    with st.spinner("Computing SHAP values..."):
        explainer = shap.TreeExplainer(rf)
        shap_vals = explainer.shap_values(X_scaled)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.summary_plot(shap_vals[1], X_scaled, feature_names=X.columns)
        st.pyplot(bbox_inches='tight')

    # Survival Analysis
    if survival_file:
        st.subheader("⏳ Kaplan-Meier Survival Analysis")
        surv = pd.read_csv(survival_file, index_col=0).loc[common]
        combined = pd.concat([expr.loc[common, top_feats['lncRNA'].iloc[0]], surv], axis=1)
        combined.columns = ['lncRNA', 'survival_time', 'status']
        combined['group'] = combined['lncRNA'] > combined['lncRNA'].median()

        try:
            cph = CoxPHFitter()
            cph.fit(combined, duration_col="survival_time", event_col="status")
            fig3 = cph.plot()
            st.pyplot(fig3.figure)
        except Exception as e:
            st.error(f"Survival model error: {e}")

    # Enrichment analysis
    st.subheader("🧬 KEGG Pathway Enrichment")
    try:
        enriched = enrichr(
            gene_list=top_feats['lncRNA'].tolist(),
            gene_sets='KEGG_2021_Human',
            organism='Human',
            outdir=None
        )
        if not enriched.results.empty:
            st.write("Top 10 KEGG Pathways:")
            st.dataframe(enriched.results[['Term', 'Adjusted P-value']].head(10))
        else:
            st.warning("No enriched pathways found.")
    except Exception as e:
        st.error(f"Enrichment failed: {e}")

    # GitHub and deployment instructions
    st.markdown("""
        ---
        ### 📦 Deployment Instructions
        1. **Install dependencies:**
           ```bash
           pip install streamlit pandas numpy scikit-learn lifelines shap seaborn matplotlib scipy gseapy
           ```
        2. **Run locally:**
           ```bash
           streamlit run lncRNA_Streamlit_App.py
           ```
        3. **Deploy to Streamlit Cloud:**
           - Push your script + a sample CSV to GitHub
           - Visit [streamlit.io/cloud](https://streamlit.io/cloud)
           - Connect GitHub repo → Deploy → Done 🎉
        4. **GitHub README Tip:**
           ```markdown
           # lncRNA-Drug Resistance Predictor
           Web app to analyze lncRNA expression and drug resistance using ML, SHAP, survival analysis, and enrichment.
           ```
    """)
else:
    st.info("Awaiting CSV file uploads...")
