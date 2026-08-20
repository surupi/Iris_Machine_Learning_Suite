import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

def create_master_dashboard(csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('ggplot')
    
    # 1. Load Data & Json Results
    df = pd.read_csv(csv_path)
    feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    
    # Create 2x2 Master Figure
    fig = plt.figure(figsize=(16, 12))
    
    # Subplot 1: Model Accuracy Leaderboard Comparison
    ax1 = fig.add_subplot(2, 2, 1)
    models = ['LDA', 'QDA', 'SVM (RBF)', 'Gaussian NB', 'MLP Neural Net', 'Logistic Reg', 'KNN', 'Random Forest']
    accuracies = [100.0, 100.0, 96.67, 96.67, 96.67, 93.33, 93.33, 93.33]
    cv_means = [97.33, 97.33, 96.00, 95.33, 94.67, 95.33, 95.33, 94.67]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax1.bar(x - width/2, accuracies, width, label='Test Acc (%)', color='#2ca02c')
    ax1.bar(x + width/2, cv_means, width, label='10-Fold CV Mean (%)', color='#1f77b4')
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_title('Top Model Accuracy Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_ylim(80, 105)
    ax1.legend()
    
    # Subplot 2: Correlation Matrix
    ax2 = fig.add_subplot(2, 2, 2)
    corr = df[feature_cols].corr().values
    im = ax2.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax2)
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels(feature_cols, rotation=30)
    ax2.set_yticklabels(feature_cols)
    for i in range(4):
        for j in range(4):
            ax2.text(j, i, f"{corr[i, j]:.2f}", ha='center', va='center', color='black', fontweight='bold')
    ax2.set_title('Feature Correlation Heatmap', fontsize=12, fontweight='bold')
    
    # Subplot 3: PCA Projection Plot
    ax3 = fig.add_subplot(2, 2, 3)
    X_scaled = StandardScaler().fit_transform(df[feature_cols])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    species_list = df['Species'].unique()
    colors = {'Iris-setosa': '#1f77b4', 'Iris-versicolor': '#ff7f0e', 'Iris-virginica': '#2ca02c'}
    
    for sp in species_list:
        mask = df['Species'] == sp
        ax3.scatter(X_pca[mask, 0], X_pca[mask, 1], label=sp, color=colors[sp], s=70, alpha=0.8)
    ax3.set_title(f'PCA Projection (Expl. Var: {sum(pca.explained_variance_ratio_)*100:.1f}%)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('PC 1')
    ax3.set_ylabel('PC 2')
    ax3.legend()
    
    # Subplot 4: Feature Importance (Random Forest)
    ax4 = fig.add_subplot(2, 2, 4)
    features = ['PetalWidth', 'PetalLength', 'SepalLength', 'SepalWidth']
    importances = [43.72, 43.15, 11.63, 1.50]
    ax4.barh(features, importances, color='#ff7f0e')
    ax4.set_xlabel('Gini Importance (%)', fontweight='bold')
    ax4.set_title('Random Forest Feature Importance', fontsize=12, fontweight='bold')
    for i, v in enumerate(importances):
        ax4.text(v + 1, i, f"{v:.2f}%", va='center', fontweight='bold')
    ax4.set_xlim(0, 50)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'master_dashboard.png'), dpi=300)
    plt.close()
    print("Master dashboard chart generated successfully.")

if __name__ == '__main__':
    csv_path = '/home/surupin@rssoftware.com/Documents/VScode/ANN-on-Iris/Iris.csv'
    out_dir = '/home/surupin@rssoftware.com/Documents/VScode/ANN-on-Iris/charts'
    create_master_dashboard(csv_path, out_dir)
