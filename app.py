# app/app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --- Page Config ---
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide",
    page_icon="📊"
)

# --- Branding Section: Banner + Logo ---
# Use a single row with 3 columns: empty, banner, empty
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    # Container for banner + top-right logo
    banner_col1, banner_col2 = st.columns([6, 1])
    with banner_col1:
        st.image("assets/images/wb1.png", width=600)  # Banner centered
    with banner_col2:
        st.image("assets/images/logo.png", width=80)   # Logo top-right

st.markdown(
    "<h1 style='text-align: center; color: #333;'>📊 Customer Segmentation Dashboard</h1>",
    unsafe_allow_html=True
)

# --- Load Data ---
@st.cache_data
def load_data():
    return pd.read_csv("assets/data/processed/rfm_clusters.csv")

df = load_data()

# --- Cluster Tagging ---
def tag_cluster(row):
    if row['Monetary'] > df['Monetary'].mean() and row['Frequency'] > df['Frequency'].mean():
        return "VIP"
    elif row['Recency'] > df['Recency'].mean():
        return "Dormant"
    else:
        return "Regular"

df['Segment'] = df.apply(tag_cluster, axis=1)
filtered_df = df.copy()

# --- Sidebar Filters ---
st.sidebar.image("assets/images/logo.png", width=100)  # Logo above filters
st.sidebar.header("🔍 Filters")
clusters = st.sidebar.multiselect(
    "Select Cluster(s):",
    options=df["Cluster"].unique(),
    default=df["Cluster"].unique()
)
filtered_df = df[df["Cluster"].isin(clusters)]

# --- KPI Cards ---
st.markdown("## 📈 Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", len(df))
col2.metric("Total Revenue", f"${df['Monetary'].sum():,.0f}")
col3.metric("Number of Clusters", df['Cluster'].nunique())

# --- Main Data Table ---
st.markdown("## 📂 Customer Dataset (with Clusters)")
st.dataframe(filtered_df)

# --- Cluster Distribution & Revenue Contribution ---
st.markdown("## 📊 Cluster Insights")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Customer Count by Cluster**")
    cluster_count = filtered_df["Cluster"].value_counts().reset_index()
    cluster_count.columns = ["Cluster", "Count"]
    fig = px.bar(
        cluster_count,
        x="Cluster", y="Count",
        color="Cluster",
        title="Number of Customers per Cluster",
        text="Count",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Shows how many customers belong to each cluster, highlighting overall distribution.")

with col2:
    st.markdown("**Revenue Contribution by Cluster**")
    revenue_by_cluster = filtered_df.groupby('Cluster')['Monetary'].sum().reset_index()
    fig = px.pie(
        revenue_by_cluster,
        names='Cluster', values='Monetary',
        title="Revenue Contribution by Cluster",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Displays total revenue contribution of each cluster, identifying high-value customers.")

# --- Scatter Plots ---
st.markdown("## 🔍 RFM Relationship Visualizations")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Recency vs Monetary**")
    fig = px.scatter(
        filtered_df,
        x="Recency", y="Monetary",
        color="Cluster", size="Frequency",
        hover_data=["CustomerID"],
        title="Recency vs Monetary",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Shows correlation between how recently customers purchased and their total spending.")

with col2:
    st.markdown("**Frequency vs Monetary**")
    fig = px.scatter(
        filtered_df,
        x='Frequency', y='Monetary',
        color='Cluster',
        title="Monetary vs Frequency",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Highlights purchasing frequency vs spending, identifying loyal customers.")

# --- Clustering Evaluation Charts ---
st.markdown("## 🧩 Clustering Evaluation")
col1, col2 = st.columns(2)

with col1:
    scaled_features = StandardScaler().fit_transform(filtered_df[['Recency', 'Frequency', 'Monetary']])
    inertia = []
    K = range(2, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_features)
        inertia.append(kmeans.inertia_)

    fig, ax = plt.subplots()
    ax.plot(K, inertia, 'bx-')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method')
    st.pyplot(fig)
    st.caption("The 'elbow' point suggests an optimal number of clusters where inertia reduction slows.")

with col2:
    silhouette_scores = []
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(scaled_features)
        silhouette_scores.append(silhouette_score(scaled_features, labels))

    fig, ax = plt.subplots()
    ax.bar(K, silhouette_scores, color='skyblue')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score by Cluster Count')
    st.pyplot(fig)
    st.caption("Higher silhouette scores indicate well-separated, meaningful clusters.")

# --- RFM Heatmap ---
st.markdown("## 🔥 Cluster Feature Averages")
rfm_cluster_means = filtered_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
fig, ax = plt.subplots(figsize=(6, 3))
sns.heatmap(rfm_cluster_means, annot=True, cmap="coolwarm", fmt=".2f")
st.pyplot(fig)
st.caption("Heatmap of average RFM metrics per cluster, providing deeper feature-level insights.")

# --- Workflow Section (centered) ---
st.subheader("📌 Workflow")
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.image("assets/images/workflow.png", caption="Customer Segmentation Pipeline", width=600)

st.markdown(
    """
**Workflow Overview:**  
The Customer Segmentation pipeline is a structured data science process designed to segment customers into actionable clusters using advanced analytics and machine learning.  
The workflow follows these steps:

1. **Data Acquisition & Integration**  
   - Aggregated transactional and demographic data from multiple sources (CRM, ERP, e-commerce platforms).  
   - Standardized data ingestion pipeline using ETL processes (extraction, transformation, and loading).  

2. **Data Cleaning & Preprocessing**  
   - Handling missing values, duplicates, and outliers using statistical techniques.  
   - Feature engineering to derive meaningful attributes (e.g., total purchase value, purchase frequency, time since last transaction).  
   - Data normalization and scaling for model stability.  

3. **Feature Engineering & RFM Modeling**  
   - Calculated **RFM metrics**:  
     - *Recency (R)*: Days since last purchase (customer activity freshness).  
     - *Frequency (F)*: Number of purchases over a time period (customer engagement).  
     - *Monetary (M)*: Total spend (customer value).  
   - Applied log transformation and scaling to reduce skewness.  

4. **Clustering & Machine Learning**  
   - Implemented **K-Means clustering** with hyperparameter tuning (elbow method, silhouette score).  
   - Validated cluster quality using metrics (e.g., Davies–Bouldin index, inertia).  
   - Segmented customers into distinct behavioral profiles.  

5. **Visualization & Reporting**  
   - Developed interactive dashboards using Streamlit and Plotly for data exploration.  
   - Created scatter plots, heatmaps, and cluster distribution charts to highlight customer group patterns.  
   - Integrated RFM segmentation insights for actionable business strategies.  

6. **Business Integration & Strategy**  
   - Mapped cluster outcomes to marketing strategies (e.g., loyalty programs for high-value customers, re-engagement campaigns for at-risk customers).  
   - Provided data-driven recommendations for personalization and revenue optimization.

> This pipeline ensures a **scalable, reproducible, and insight-driven segmentation system** for advanced customer analytics.
""",
    unsafe_allow_html=True
)

# --- Download Button ---
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Segmented Data",
    data=csv,
    file_name='segmented_customers.csv',
    mime='text/csv'
)

# --- Insights ---
st.markdown("## 🔑 Quick Insights")
st.info(
    """
    - **VIP Cluster**: High-value customers, target with loyalty programs.  
    - **Dormant Cluster**: Customers at risk, re-engage with win-back campaigns.  
    - **Regular Cluster**: Stable buyers, potential to upsell.  
    """
)

# --- Footer ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Developed by <a href='https://github.com/akbarnaeem' target='_blank'>Akbar Naeem</a></p>",
    unsafe_allow_html=True
)
