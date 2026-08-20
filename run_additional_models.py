import os
import json
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Additional Model Imports
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier, PassiveAggressiveClassifier, Perceptron
from sklearn.neighbors import NearestCentroid, RadiusNeighborsClassifier

def run_extended_models(csv_path):
    df = pd.read_csv(csv_path)
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])
        
    feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    target_col = 'Species'
    
    X = df[feature_cols]
    y = df[target_col]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    additional_models = {
        'Gaussian Naive Bayes': GaussianNB(),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
        'Quadratic Discriminant Analysis (QDA)': QuadraticDiscriminantAnalysis(),
        'Extra Trees Classifier': ExtraTreesClassifier(n_estimators=100, random_state=42),
        'AdaBoost Classifier': AdaBoostClassifier(n_estimators=50, random_state=42),
        'Hist Gradient Boosting': HistGradientBoostingClassifier(random_state=42),
        'Ridge Classifier': RidgeClassifier(random_state=42),
        'Passive Aggressive': PassiveAggressiveClassifier(random_state=42),
        'Perceptron': Perceptron(random_state=42),
        'Nearest Centroid': NearestCentroid(),
        'Radius Neighbors': RadiusNeighborsClassifier(radius=1.5)
    }
    
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = {}
    
    for name, model in additional_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=cv, scoring='accuracy')
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        results[name] = {
            'test_accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'cv_accuracy_mean': float(cv_scores.mean()),
            'cv_accuracy_std': float(cv_scores.std()),
            'confusion_matrix': cm
        }
        
    return results

if __name__ == '__main__':
    csv_path = '/home/surupin@rssoftware.com/Documents/VScode/ANN-on-Iris/Iris.csv'
    res = run_extended_models(csv_path)
    
    out_dir = '/home/surupin@rssoftware.com/Documents/VScode/ANN-on-Iris'
    with open(os.path.join(out_dir, 'extended_model_results.json'), 'w') as f:
        json.dump(res, f, indent=2)
        
    print("Extended models benchmarking completed successfully.")
