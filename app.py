import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Classifier Models
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="Iris ML & Analytics Suite",
    page_icon="🌸",
    layout="wide"
)

# Custom Sidebar Spacing & Font Styling CSS
st.markdown("""
    <style>
    /* Increase font size and line height for sidebar title and labels */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        font-size: 1.15rem !important;
        font-weight: 500 !important;
        margin-bottom: 8px !important;
    }
    
    /* Radio options font size and padding */
    [data-testid="stSidebar"] div[role="radiogroup"] label {
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        padding-top: 8px !important;
        padding-bottom: 8px !important;
        margin-bottom: 10px !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label p {
        font-size: 1.1rem !important;
        font-weight: 500 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('Iris.csv')
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])
    return df

df = load_data()
feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
species_list = df['Species'].unique()

# Title
st.title("🌸 Iris Dataset Analysis & 18-Model ML Suite")
st.markdown("An interactive web application for exploratory data analysis, statistical testing, PCA/LDA projections, model benchmarking, and real-time species classification.")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", [
    "📊 Data Explorer & Statistics",
    "📐 Dimensionality Reduction (PCA & LDA)",
    "🏆 18-Model Leaderboard",
    "🎯 Live Flower Predictor"
])

# Page 1: Data Explorer & Statistics
if page == "📊 Data Explorer & Statistics":
    st.header("📊 Exploratory Data Analysis & Statistics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
    with col2:
        st.subheader("Summary Statistics")
        st.dataframe(df.describe().T[['mean', 'std', 'min', '50%', 'max']], use_container_width=True)
        
    st.divider()
    
    st.subheader("Feature Distributions by Species")
    selected_feature = st.selectbox("Select Feature for Distribution Boxplot:", feature_cols)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    data_to_plot = [df[df['Species'] == sp][selected_feature] for sp in species_list]
    bplot = ax.boxplot(data_to_plot, patch_artist=True, tick_labels=species_list)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_title(f"{selected_feature} Distribution", fontsize=12, fontweight='bold')
    st.pyplot(fig)

# Page 2: Dimensionality Reduction (PCA & LDA)
elif page == "📐 Dimensionality Reduction (PCA & LDA)":
    st.header("📐 Dimensionality Reduction (PCA & LDA)")
    
    X_scaled = StandardScaler().fit_transform(df[feature_cols])
    le = LabelEncoder()
    y_encoded = le.fit_transform(df['Species'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Principal Component Analysis (PCA)")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = {'Iris-setosa': '#1f77b4', 'Iris-versicolor': '#ff7f0e', 'Iris-virginica': '#2ca02c'}
        for sp in species_list:
            mask = df['Species'] == sp
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=sp, color=colors[sp], s=70, alpha=0.8)
        ax.set_title(f"PCA 2D Scatter (PC1: {pca.explained_variance_ratio_[0]*100:.1f}%, PC2: {pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.legend()
        st.pyplot(fig)
        
    with col2:
        st.subheader("Linear Discriminant Analysis (LDA)")
        lda = LDA(n_components=2)
        X_lda = lda.fit_transform(X_scaled, y_encoded)
        
        fig, ax = plt.subplots(figsize=(7, 5))
        for sp in species_list:
            mask = df['Species'] == sp
            ax.scatter(X_lda[mask, 0], X_lda[mask, 1], label=sp, color=colors[sp], s=70, alpha=0.8)
        ax.set_title(f"LDA 2D Scatter (LD1: {lda.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_xlabel("LD 1")
        ax.set_ylabel("LD 2")
        ax.legend()
        st.pyplot(fig)

# Page 3: 18-Model Leaderboard
elif page == "🏆 18-Model Leaderboard":
    st.header("🏆 Supervised Machine Learning Benchmark Leaderboard")
    
    leaderboard_data = {
        "Rank": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        "Model Name": [
            "Linear Discriminant Analysis (LDA)", "Quadratic Discriminant Analysis (QDA)",
            "SVM (RBF Kernel)", "Gaussian Naive Bayes", "MLP Neural Network",
            "Logistic Regression", "K-Nearest Neighbors", "Random Forest",
            "Extra Trees Classifier", "AdaBoost Classifier", "Gradient Boosting",
            "Hist Gradient Boosting", "Decision Tree", "Passive Aggressive",
            "Perceptron", "Radius Neighbors", "Nearest Centroid", "Ridge Classifier"
        ],
        "Category": [
            "Discriminant", "Discriminant", "Non-Linear Kernel", "Probabilistic",
            "Neural Network", "Linear Model", "Distance-Based", "Ensemble",
            "Ensemble", "Ensemble", "Ensemble", "Ensemble",
            "Tree-Based", "Online Learning", "Linear Neural", "Distance-Based",
            "Centroid-Based", "Linear Regularized"
        ],
        "Test Accuracy (%)": [100.0, 100.0, 96.67, 96.67, 96.67, 93.33, 93.33, 93.33, 93.33, 93.33, 90.0, 90.0, 90.0, 90.0, 86.67, 86.67, 83.33, 76.67],
        "10-Fold CV Mean (%)": [97.33, 97.33, 96.00, 95.33, 94.67, 95.33, 95.33, 94.67, 95.33, 95.33, 94.00, 94.00, 93.33, 85.33, 87.33, 86.67, 86.00, 82.67]
    }
    
    lb_df = pd.DataFrame(leaderboard_data)
    st.dataframe(lb_df, use_container_width=True, hide_index=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(lb_df['Model Name'][::-1], lb_df['Test Accuracy (%)'][::-1], color='#2ca02c')
    ax.set_xlabel("Test Accuracy (%)")
    ax.set_title("18-Model Accuracy Leaderboard Comparison")
    ax.set_xlim(60, 105)
    st.pyplot(fig)

# Page 4: Live Flower Predictor
elif page == "🎯 Live Flower Predictor":
    st.header("🎯 Real-Time Flower Species Predictor")
    st.write("Adjust the sliders below to input flower dimensions and classify its species in real time.")
    
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.8, step=0.1)
        sepal_width = st.slider("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.0, step=0.1)
    with col2:
        petal_length = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, value=3.8, step=0.1)
        petal_width = st.slider("Petal Width (cm)", min_value=0.1, max_value=2.5, value=1.2, step=0.1)
        
    model_choice = st.selectbox("Select Classifier Model:", [
        "Linear Discriminant Analysis (LDA)",
        "Quadratic Discriminant Analysis (QDA)",
        "SVM (RBF Kernel)",
        "Random Forest",
        "K-Nearest Neighbors"
    ])
    
    # Train Selected Model
    X = df[feature_cols]
    y = df['Species']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models_dict = {
        "Linear Discriminant Analysis (LDA)": LinearDiscriminantAnalysis(),
        "Quadratic Discriminant Analysis (QDA)": QuadraticDiscriminantAnalysis(),
        "SVM (RBF Kernel)": SVC(kernel='rbf', probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
    }
    
    model = models_dict[model_choice]
    model.fit(X_scaled, y_encoded)
    
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_features)
    
    pred_idx = model.predict(input_scaled)[0]
    pred_species = le.inverse_transform([pred_idx])[0]
    
    st.subheader(f"Predicted Species: **{pred_species}** 🌺")
    
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(input_scaled)[0]
        prob_df = pd.DataFrame({'Species': le.classes_, 'Probability (%)': [p * 100 for p in probs]})
        st.bar_chart(prob_df.set_index('Species'))
