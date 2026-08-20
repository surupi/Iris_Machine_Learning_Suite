import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def generate_charts(csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('ggplot')
    
    df = pd.read_csv(csv_path)
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])
        
    feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    species_list = df['Species'].unique()
    colors = {'Iris-setosa': '#1f77b4', 'Iris-versicolor': '#ff7f0e', 'Iris-virginica': '#2ca02c'}
    
    # 1. Feature Boxplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, col in enumerate(feature_cols):
        ax = axes[i // 2, i % 2]
        data_to_plot = [df[df['Species'] == sp][col] for sp in species_list]
        bplot = ax.boxplot(data_to_plot, patch_artist=True, tick_labels=species_list)
        for patch, sp in zip(bplot['boxes'], species_list):
            patch.set_facecolor(colors[sp])
        ax.set_title(f'{col} Distribution by Species', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_boxplots.png'), dpi=300)
    plt.close()
    
    # 2. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    corr = df[feature_cols].corr().values
    im = plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.xticks(range(4), feature_cols, rotation=45)
    plt.yticks(range(4), feature_cols)
    for i in range(4):
        for j in range(4):
            plt.text(j, i, f"{corr[i, j]:.2f}", ha='center', va='center', color='black')
    plt.title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300)
    plt.close()
    
    # 3. PCA Projection
    X_scaled = StandardScaler().fit_transform(df[feature_cols])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(9, 7))
    for sp in species_list:
        mask = df['Species'] == sp
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=sp, color=colors[sp], s=80, alpha=0.8)
    plt.title(f'PCA Projection (PC1: {pca.explained_variance_ratio_[0]*100:.1f}%, PC2: {pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=14, fontweight='bold')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_projection.png'), dpi=300)
    plt.close()
    
    # 4. LDA Projection
    le = LabelEncoder()
    y_encoded = le.fit_transform(df['Species'])
    lda = LDA(n_components=2)
    X_lda = lda.fit_transform(X_scaled, y_encoded)
    
    plt.figure(figsize=(9, 7))
    for sp in species_list:
        mask = df['Species'] == sp
        plt.scatter(X_lda[mask, 0], X_lda[mask, 1], label=sp, color=colors[sp], s=80, alpha=0.8)
    plt.title('Linear Discriminant Analysis (LDA) 2D Projection', fontsize=14, fontweight='bold')
    plt.xlabel('Linear Discriminant 1')
    plt.ylabel('Linear Discriminant 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lda_projection.png'), dpi=300)
    plt.close()
    
    print("Charts generated successfully.")

if __name__ == '__main__':
    csv_path = '/home/surupin@rssoftware.com/Documents/VScode/ANN-on-Iris/Iris.csv'
    out_dir = '/home/surupin@rssoftware.com/.gemini/antigravity/brain/a88353a6-b004-4d05-b3af-9b6edee8aed4/scratch/charts'
    generate_charts(csv_path, out_dir)
