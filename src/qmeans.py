import streamlit as st
import matplotlib.pyplot as pt
import qlib as q
import numpy as np

st.set_page_config(layout='wide')

st.title("Quantum K-Means Clustering")
st.write("Adjust the parameters below:")
n, k, std = st.columns(3)

with n:
    n_samples = st.slider("Number of Samples", 100, 1000, 500)
with k:
    n_clusters = st.slider("Number of Clusters", 2, 10, 3)
with std:
    cluster_std = st.slider("Cluster Standard Deviation", 0.1, 2.0, 1.0)

generate_button = st.button("Generate Plots")

if generate_button:
    points, centers = q.generate_data(n_samples, n_clusters, cluster_std)
    points = q.preprocess(points)
    centroids = points[np.random.randint(points.shape[0],size=n_clusters),:]
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Initial Data Points")
        fig1, ax1 = pt.subplots()
        ax1.set_facecolor('black')
        ax1.scatter(points[:, 0], points[:, 1], c='black', s=20)
        ax1.set_title("Generated Data")
        ax1.set_axis_off()
        st.pyplot(fig1)
 
    with col2:
        st.subheader("Quantum K-Means Progress")
        placeholder = st.empty()

    for i in range(6):
        fig2, ax2 = pt.subplots()
        centers = q.find_nearest_neighbour(points, centroids)
        ax2.scatter(points[:, 0], points[:, 1], c=centers, cmap='viridis', s=20)
        ax2.set_axis_off()
        ax2.set_title(f"Iteration {i + 1}")
        placeholder.pyplot(fig2)
        centroids = q.find_centroids(points, centers)
