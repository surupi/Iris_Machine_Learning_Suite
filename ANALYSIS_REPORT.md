# Comprehensive Iris Dataset Analysis Walkthrough Report

An end-to-end analytical study of the **Iris Dataset (`Iris.csv`)**, covering exploratory data analysis, hypothesis testing, dimensionality reduction, multi-model supervised classification, and unsupervised clustering.

---

## 1. Executive Summary
- **Dataset Size**: 150 instances, 4 continuous features (`SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`), 1 target label (`Species`).
- **Class Balance**: Perfectly balanced across 3 species (50 *Iris-setosa*, 50 *Iris-versicolor*, 50 *Iris-virginica*).
- **Data Quality**: 0 missing values, 3 duplicate feature rows.
- **Top Classifier**: **Linear Discriminant Analysis (LDA)** and **Quadratic Discriminant Analysis (QDA)** achieved **100.0% test accuracy** and **97.33% 10-fold CV mean accuracy**.
- **Clustering Benchmark**: K-Means clustering achieved an **Adjusted Rand Index (ARI) of 0.62** and **Normalized Mutual Information (NMI) of 0.66** against ground truth labels.

![Master Dashboard](/home/surupin@rssoftware.com/.gemini/antigravity/brain/a88353a6-b004-4d05-b3af-9b6edee8aed4/master_dashboard.png)

---

## 2. Feature Distributions & Bivariate Correlation

![Feature Distributions Boxplot](/home/surupin@rssoftware.com/.gemini/antigravity/brain/a88353a6-b004-4d05-b3af-9b6edee8aed4/feature_boxplots.png)

### Summary Statistics

| Feature | Overall Mean ± Std | Iris-setosa Mean | Iris-versicolor Mean | Iris-virginica Mean |
| :--- | :--- | :--- | :--- | :--- |
| **SepalLengthCm** | 5.84 ± 0.83 cm | 5.01 cm | 5.94 cm | 6.59 cm |
| **SepalWidthCm** | 3.05 ± 0.43 cm | 3.42 cm | 2.77 cm | 2.97 cm |
| **PetalLengthCm** | 3.76 ± 1.76 cm | 1.46 cm | 4.26 cm | 5.55 cm |
| **PetalWidthCm** | 1.20 ± 0.76 cm | 0.24 cm | 1.33 cm | 2.03 cm |

### Correlation Heatmap

![Correlation Heatmap](/home/surupin@rssoftware.com/.gemini/antigravity/brain/a88353a6-b004-4d05-b3af-9b6edee8aed4/correlation_heatmap.png)

- **PetalLengthCm & PetalWidthCm** show an extremely high linear correlation (**r = 0.96**).
- **SepalLengthCm** correlates strongly with **PetalLengthCm (r = 0.87)** and **PetalWidthCm (r = 0.82)**.
- **SepalWidthCm** shows a weak negative correlation with all other 3 features.

---

## 3. Statistical Testing & Significance

### 1. Normality Assessment (Shapiro-Wilk Test)
- `SepalLengthCm` and `SepalWidthCm` follow a normal distribution across species ($p > 0.05$).
- `PetalWidthCm` for *Iris-setosa* deviates significantly from normality ($p = 1.85 \times 10^{-6}$).

### 2. Analysis of Variance (One-Way ANOVA & Kruskal-Wallis)
All features show statistically significant differences across species ($p < 10^{-15}$):

| Feature | ANOVA F-Statistic | ANOVA p-value | Kruskal-Wallis H |
| :--- | :--- | :--- | :--- |
| **PetalLengthCm** | **1179.03** | **$3.05 \times 10^{-91}$** | **130.41** |
| **PetalWidthCm** | **959.32** | **$4.38 \times 10^{-85}$** | **131.09** |
| **SepalLengthCm** | 119.26 | $1.67 \times 10^{-31}$ | 96.94 |
| **SepalWidthCm** | 47.36 | $1.33 \times 10^{-16}$ | 62.49 |

---

## 4. Dimensionality Reduction (PCA & LDA)

![PCA Projection](/home/surupin@rssoftware.com/.gemini/antigravity/brain/a88353a6-b004-4d05-b3af-9b6edee8aed4/pca_projection.png)

![LDA Projection](/home/surupin@rssoftware.com/.gemini/antigravity/brain/a88353a6-b004-4d05-b3af-9b6edee8aed4/lda_projection.png)

- **Principal Component Analysis (PCA)**:
  - **PC1** explains **72.77%** of the variance.
  - **PC2** explains **23.03%** of the variance.
  - Cumulative variance of top 2 components = **95.80%**.
- **Linear Discriminant Analysis (LDA)**:
  - **LD1** accounts for **99.15%** of class separability, demonstrating complete linear separation of *Iris-setosa* from the remaining two classes.

---

## 5. Supervised Model Benchmark Results

Master benchmark comparing 18 machine learning algorithm architectures on a standard scaled 80/20 train-test split with 10-Fold Stratified Cross-Validation:

| Model Category | Model Name | Test Accuracy | Precision | Recall | F1-Score | 10-Fold CV Mean (± Std) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Generative / Discriminant** | **Linear Discriminant Analysis (LDA)** | **100.0%** | **1.000** | **1.000** | **1.000** | **97.33% (±4.4%)** |
| **Generative / Discriminant** | **Quadratic Discriminant Analysis (QDA)** | **100.0%** | **1.000** | **1.000** | **1.000** | **97.33% (±4.4%)** |
| **Non-Linear Kernel** | **SVM (RBF Kernel)** | **96.67%** | **0.970** | **0.967** | **0.967** | **96.00% (±4.4%)** |
| **Neural Network** | **MLP Neural Network** | **96.67%** | **0.970** | **0.967** | **0.967** | **94.67% (±5.8%)** |
| **Probabilistic** | **Gaussian Naive Bayes** | **96.67%** | **0.970** | **0.967** | **0.967** | **95.33% (±5.2%)** |
| **Linear Model** | **Logistic Regression** | **93.33%** | 0.933 | 0.933 | 0.933 | 95.33% (±5.2%) |
| **Distance-Based** | **K-Nearest Neighbors (KNN)** | **93.33%** | 0.944 | 0.933 | 0.933 | 95.33% (±5.2%) |
| **Ensemble** | **Random Forest** | **93.33%** | 0.933 | 0.933 | 0.933 | 94.67% (±5.0%) |
| **Ensemble** | **Extra Trees Classifier** | **93.33%** | 0.933 | 0.933 | 0.933 | 95.33% (±5.2%) |
| **Ensemble** | **AdaBoost Classifier** | **93.33%** | 0.933 | 0.933 | 0.933 | 95.33% (±5.2%) |
| **Ensemble** | **Gradient Boosting** | **90.00%** | 0.902 | 0.900 | 0.900 | 94.00% (±4.7%) |
| **Ensemble** | **Hist Gradient Boosting** | **90.00%** | 0.902 | 0.900 | 0.900 | 94.00% (±4.7%) |
| **Tree-Based** | **Decision Tree** | **90.00%** | 0.902 | 0.900 | 0.900 | 93.33% (±5.2%) |
| **Online Learning** | **Passive Aggressive Classifier** | **90.00%** | 0.902 | 0.900 | 0.900 | 85.33% (±10.7%) |
| **Linear Neural** | **Perceptron** | **86.67%** | 0.867 | 0.867 | 0.862 | 87.33% (±9.2%) |
| **Distance-Based** | **Radius Neighbors Classifier** | **86.67%** | 0.875 | 0.867 | 0.865 | 86.67% (±9.4%) |
| **Centroid-Based** | **Nearest Centroid** | **83.33%** | 0.835 | 0.833 | 0.833 | 86.00% (±8.7%) |
| **Linear Model** | **Ridge Classifier** | **76.67%** | 0.777 | 0.767 | 0.761 | 82.67% (±10.8%) |

### Gini Feature Importance (Random Forest)
1. **PetalWidthCm**: 43.72%
2. **PetalLengthCm**: 43.15%
3. **SepalLengthCm**: 11.63%
4. **SepalWidthCm**: 1.50%

---

## 6. Unsupervised Clustering Evaluation

Unsupervised models evaluated against true species labels:

| Clustering Algorithm | Adjusted Rand Index (ARI) | Normalized Mutual Info (NMI) |
| :--- | :--- | :--- |
| **K-Means ($k=3$)** | 0.6201 | 0.6595 |
| **Agglomerative Clustering ($k=3$)** | 0.6153 | 0.6755 |

---

## 7. Conclusions & Takeaways
1. **Petal dimensions (`PetalLengthCm` and `PetalWidthCm`)** provide over **86%** of total predictive importance for distinguishing flower species.
2. *Iris-setosa* is linearly separable from the other two species.
3. *Iris-versicolor* and *Iris-virginica* have overlapping feature boundaries, which non-linear decision boundaries (SVM RBF, Neural Networks) handle best.
