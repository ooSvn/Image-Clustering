
# Flower Image Clustering using VGG16 and Scikit-learn

---

## 1. Goal
The primary objective of this project is to perform unsupervised clustering on a dataset of flower images. By using advanced feature extraction techniques and various clustering algorithms, we aim to group similar images together without any prior knowledge of their true labels. The performance of the models is then evaluated against a hidden set of ground-truth labels.

---

## 2. Data
- **Source**: `./flower_images/` directory containing JPG files.
- **Data Shape**: 170 images of various flowers.
- **Labels**: Ground-truth labels for evaluation are provided in `flower_labels.csv`.

**Feature Extraction**  
Instead of using raw pixel values, which are high-dimensional and less informative, we leverage **VGG16**, a pre-trained Convolutional Neural Network (CNN). VGG16 extracts rich, hierarchical features from each image, which are then flattened into a vector. This approach has several advantages:  
*   **Hierarchical Representation**: Captures complex patterns from edges to semantic shapes.  
*   **Dimensionality Reduction**: Reduces the feature space significantly.  
*   **Transfer Learning**: Uses knowledge from a large dataset (ImageNet), improving performance on our smaller dataset.

---

## 3. Pipeline Overview
1.  **Load Images**: Read all images from the `./flower_images/` directory.
2.  **Pre-processing**: Resize images to 224x224 and convert to RGB format.
3.  **Feature Extraction**: Use the VGG16 model (without its final classification layer) to generate a feature vector for each image.
4.  **Feature Engineering**: Apply `Normalizer` and `PCA` to the extracted features to prepare them for clustering.
5.  **Clustering**: Experiment with two popular unsupervised algorithms: K-Means and DBSCAN.
6.  **Evaluation**: Assess clustering quality using internal and external metrics:
    - **Internal**: `Silhouette Score` (measures cluster compactness).
    - **External**: `Homogeneity`, `Completeness`, and `V-Measure` (compare clusters to ground-truth labels).

---

## 4. Algorithms & Key Results
### K-Means
- **Methodology**: Partitions data into a pre-defined number of clusters. We used the **Elbow Method** and **Silhouette Analysis** to determine the optimal number of clusters (`k`). Both methods pointed to a `k` of 9.
- **Performance**:
    - `Homogeneity Score`: 0.55
    - `Completeness Score`: 0.62
    - `V-Measure Score`: 0.58
    - `Silhouette Score`: 0.026
- **Visualization**: A 2D PCA plot showed the clusters were visually distinct, but with some overlap, explaining the modest silhouette score.

### DBSCAN
- **Methodology**: Density-based spatial clustering that groups together points that are closely packed together, marking as outliers points that lie alone in low-density regions. The `eps` and `min_samples` parameters were tuned using a k-distance graph, and `cosine_distances` were used for the `metric` parameter.
- **Performance**:
    - `Homogeneity Score`: 0.51
    - `Completeness Score`: 0.60
    - `V-Measure Score`: 0.55
    - `Silhouette Score`: 0.13
- **Visualization**: Similar to K-Means, the 2D PCA plot showed visually separated clusters, with some noise points identified correctly.

### Comparative Analysis
Both K-Means and DBSCAN produced similar results, with K-Means slightly outperforming DBSCAN on external metrics. The V-Measure scores around 0.55-0.58 indicate a decent level of clustering, but there is still room for improvement. The low silhouette scores suggest that the clusters, while homogeneous, are not perfectly compact or well-separated, which is common in complex, high-dimensional data like images.

---
