# streamlit_app.py

import streamlit as st
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import zipfile

st.set_page_config(page_title="DBSCAN Clustering App", layout="wide")

st.title("🔍 DBSCAN Clustering on Credit Card Dataset")

# Upload ZIP file
uploaded_file = st.file_uploader("Upload a zipped CSV file (like archive.zip)", type=["zip"])

if uploaded_file:
    try:
        with zipfile.ZipFile(uploaded_file, 'r') as z:
            # Assumes only one CSV file inside the ZIP
            file_name = z.namelist()[0]
            with z.open(file_name) as f:
                df = pd.read_csv(f)

        st.success("✅ File uploaded and extracted successfully!")

        st.subheader("📊 Raw Data Preview")
        st.dataframe(df.head())

        # Feature selection
        if 'Time' in df.columns and 'Class' in df.columns:
            X = df.drop(columns=['Time', 'Class'])
        else:
            X = df.copy()

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # User inputs for DBSCAN
        st.sidebar.header("🔧 DBSCAN Parameters")
        eps = st.sidebar.slider("Epsilon (eps)", 0.1, 10.0, 2.5, 0.1)
        min_samples = st.sidebar.slider("Minimum Samples", 1, 50, 15)

        # Apply DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
        db.fit(X_scaled)
        labels = db.labels_

        # Core mask and cluster metrics
        mask_core = labels != -1
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)

        if len(np.unique(labels[mask_core])) > 1:
            sil = silhouette_score(X_scaled[mask_core], labels[mask_core])
        else:
            sil = np.nan

        st.subheader("📈 DBSCAN Clustering Results")
        st.write(f"**Estimated clusters:** {n_clusters}")
        st.write(f"**Noise points:** {n_noise}")
        st.write(f"**Silhouette Score (core only):** {sil:.4f}" if not np.isnan(sil) else "Silhouette Score: Not enough clusters")

        # Add clustering labels to data
        df['Cluster'] = labels
        st.subheader("📁 Clustered Data Sample")
        st.dataframe(df.head())

        # Optional: Download clustered result
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Clustered Data as CSV", csv, "clustered_output.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error processing the file: {e}")
else:
    st.info("📤 Please upload a ZIP file containing your CSV data.")
