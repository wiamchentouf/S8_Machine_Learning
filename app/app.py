import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

# --- Configuration de la page ---
st.set_page_config(page_title="Projet ML - S8", layout="wide")

# --- Création de données synthétiques (pour éviter l'erreur data.csv) ---
@st.cache_data
def get_data():
    X, y = make_classification(n_samples=200, n_features=5, n_classes=2, random_state=42)
    df = pd.DataFrame(X, columns=[f'Variable_{i}' for i in range(5)])
    df['target'] = y
    return df, X, y

df, X, y = get_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Menu latéral ---
st.sidebar.title("Sommaire")
menu = ["Accueil", "KNN", "LDA/QDA", "PCA", "SVM", "Decision Tree"]
choice = st.sidebar.selectbox("Choisissez une méthode", menu)

# --- Fonction pour les résultats ---
def afficher_resultats(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"### Précision (Accuracy) : {acc:.2%}")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    st.pyplot(fig)

# --- Contenu ---
if choice == "Accueil":
    st.title("Projet Machine Learning - S8")
    st.write("Bienvenue. Utilisez le menu pour explorer les modèles.")

elif choice == "KNN":
    st.title("K-Nearest Neighbors (KNN)")
    model = KNeighborsClassifier().fit(X_train, y_train)
    afficher_resultats(model, X_test, y_test)

elif choice == "LDA/QDA":
    st.title("LDA")
    model = LinearDiscriminantAnalysis().fit(X_train, y_train)
    afficher_resultats(model, X_test, y_test)

elif choice == "PCA":
    st.title("Analyse en Composantes Principales (PCA)")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    fig, ax = plt.subplots()
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
    st.pyplot(fig)

elif choice == "SVM":
    st.title("Support Vector Machines (SVM)")
    model = SVC(kernel='linear').fit(X_train, y_train)
    afficher_resultats(model, X_test, y_test)

elif choice == "Decision Tree":
    st.title("Arbre de Décision")
    model = DecisionTreeClassifier().fit(X_train, y_train)
    afficher_resultats(model, X_test, y_test)
