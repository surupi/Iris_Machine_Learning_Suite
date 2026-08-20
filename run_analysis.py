import os
import json
import pandas as pd
import numpy as np
import scipy.stats as stats
# statsmodels optional imports removed for environment compatibility

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, adjusted_rand_score, normalized_mutual_info_score
)

# Classifier Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering

def run_analysis(csv_path):
    results = {}
    
    # 1. Load Data
    df = pd.read_csv(csv_path)
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])
        
    feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    target_col = 'Species'
    
    # 2. EDA & Basic Info
    results['dataset_shape'] = df.shape
    results['missing_values'] = df.isnull().sum().to_dict()
    results['duplicate_rows'] = int(df.duplicated().sum())
    results['class_counts'] = df[target_col].value_counts().to_dict()
    
    # Overall summary statistics
    results['overall_stats'] = df[feature_cols].describe().to_dict()
    
    # Species-wise summary statistics
    species_stats = {}
    for sp in df[target_col].unique():
        species_stats[sp] = df[df[target_col] == sp][feature_cols].describe().to_dict()
    results['species_stats'] = species_stats
    
    # Skewness & Kurtosis
    results['skewness'] = df[feature_cols].skew().to_dict()
    results['kurtosis'] = df[feature_cols].kurtosis().to_dict()
    
    # Correlation matrix
    results['correlation_matrix'] = df[feature_cols].corr().to_dict()
    
    # 3. Statistical Testing
    # Shapiro-Wilk test for normality per feature per species
    shapiro_results = {}
    for sp in df[target_col].unique():
        shapiro_results[sp] = {}
        sp_df = df[df[target_col] == sp]
        for col in feature_cols:
            stat, p_val = stats.shapiro(sp_df[col])
            shapiro_results[sp][col] = {'stat': float(stat), 'p_value': float(p_val)}
    results['shapiro_test'] = shapiro_results
    
    # ANOVA test across species for each feature
    anova_results = {}
    for col in feature_cols:
        groups = [group[col].values for name, group in df.groupby(target_col)]
        f_stat, p_val = stats.f_oneway(*groups)
        kw_stat, kw_p = stats.kruskal(*groups)
        
        anova_results[col] = {
            'anova_f_stat': float(f_stat),
            'anova_p_val': float(p_val),
            'kruskal_stat': float(kw_stat),
            'kruskal_p_val': float(kw_p)
        }
    results['anova_results'] = anova_results
    
    # 4. Dimensionality Reduction & Feature Selection
    X = df[feature_cols]
    y = df[target_col]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=4)
    pca.fit(X_scaled)
    results['pca_explained_variance_ratio'] = pca.explained_variance_ratio_.tolist()
    results['pca_cumulative_variance'] = np.cumsum(pca.explained_variance_ratio_).tolist()
    
    # LDA
    lda = LDA(n_components=2)
    X_lda = lda.fit_transform(X_scaled, y_encoded)
    results['lda_explained_variance_ratio'] = lda.explained_variance_ratio_.tolist()
    
    # 5. Supervised Learning Models
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'MLP Neural Network': MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
    }
    
    model_eval = {}
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=cv, scoring='accuracy')
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        eval_dict = {
            'test_accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'cv_accuracy_mean': float(cv_scores.mean()),
            'cv_accuracy_std': float(cv_scores.std()),
            'confusion_matrix': cm
        }
        
        if hasattr(model, 'feature_importances_'):
            eval_dict['feature_importances'] = dict(zip(feature_cols, model.feature_importances_.tolist()))
            
        model_eval[name] = eval_dict
        
    results['supervised_models'] = model_eval
    
    # 6. Unsupervised Clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    agg = AgglomerativeClustering(n_clusters=3)
    agg_labels = agg.fit_predict(X_scaled)
    
    results['clustering'] = {
        'kmeans': {
            'ari': float(adjusted_rand_score(y_encoded, kmeans_labels)),
            'nmi': float(normalized_mutual_info_score(y_encoded, kmeans_labels))
        },
        'agglomerative': {
            'ari': float(adjusted_rand_score(y_encoded, agg_labels)),
            'nmi': float(normalized_mutual_info_score(y_encoded, agg_labels))
        }
    }
    
    return results

if __name__ == '__main__':
    csv_path = '/home/surupin@rssoftware.com/Documents/VScode/ANN-on-Iris/Iris.csv'
    res = run_analysis(csv_path)
    
    out_dir = '/home/surupin@rssoftware.com/.gemini/antigravity/brain/a88353a6-b004-4d05-b3af-9b6edee8aed4/scratch'
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, 'analysis_results.json'), 'w') as f:
        json.dump(res, f, indent=2)
        
    print("Analysis script completed successfully.")
