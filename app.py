import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.decomposition import PCA

# --- Configuration ---
st.set_page_config(page_title="Projet ML - S8", layout="wide")

@st.cache_data
def get_data():
    X, y = make_classification(n_samples=300, n_features=5, n_classes=2, random_state=42)
    return X, y

X, y = get_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Fonction pour afficher les résultats et le graphique ---
def afficher_model(model, X_test, y_test, nom_model):
    y_pred = model.predict(X_test)
    st.write(f"### Résultats pour {nom_model}")
    st.write(f"**Précision :** {accuracy_score(y_test, y_pred):.2%}")
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    plt.title("Matrice de Confusion")
    st.pyplot(fig)

# --- Menu ---
st.sidebar.title("Navigation")
menu = ["Accueil", "KNN", "LDA", "PCA", "SVM", "Decision Tree"]
choice = st.sidebar.selectbox("Sélectionnez l'algorithme", menu)

# --- Logique ---
if choice == "Accueil":
    st.title("Interface de Démonstration : Machine Learning")
    st.write("Bienvenue. Choisissez un algorithme dans le menu à gauche pour voir les résultats et les graphiques.")

elif choice == "KNN":
    model = KNeighborsClassifier().fit(X_train, y_train)
    afficher_model(model, X_test, y_test, "KNN")

elif choice == "LDA":
    model = LinearDiscriminantAnalysis().fit(X_train, y_train)
    afficher_model(model, X_test, y_test, "LDA")

elif choice == "PCA":
    st.title("Analyse en Composantes Principales (PCA)")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    fig, ax = plt.subplots()
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolors='k')
    plt.title("Projection PCA (2D)")
    st.pyplot(fig)

elif choice == "SVM":
    model = SVC().fit(X_train, y_train)
    afficher_model(model, X_test, y_test, "SVM")

elif choice == "Decision Tree":
    st.title("Arbre de Décision")
    model = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)
    afficher_model(model, X_test, y_test, "Arbre de Décision")
    
    st.write("### Visualisation de l'Arbre")
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_tree(model, filled=True, ax=ax)
    st.pyplot(fig)
