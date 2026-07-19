import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

# --- Configuration de la page ---
st.set_page_config(page_title="Projet ML - S8", layout="wide")

# --- Chargement des données ---
@st.cache_data
def load_data():
    # Remplacez 'data.csv' par le nom réel de votre fichier
    return pd.read_csv("data.csv") 

try:
    df = load_data()
    X = df.drop('target', axis=1) # Assurez-vous que 'target' est le nom de votre colonne cible
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
except:
    st.error("Erreur : Fichier 'data.csv' non trouvé ou format incorrect.")
    st.stop()

# --- Menu latéral ---
st.sidebar.title("Sommaire")
menu = ["Accueil", "KNN", "LDA/QDA", "PCA", "SVM", "Decision Tree"]
choice = st.sidebar.selectbox("Choisissez une méthode", menu)

# --- Fonction pour afficher les résultats ---
def afficher_resultats(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    st.write(f"### Précision (Accuracy) : {acc:.2%}")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matrice de Confusion")
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    st.pyplot(fig)

# --- Contenu principal ---
if choice == "Accueil":
    st.title("Projet Machine Learning - S8")
    st.write("Bienvenue sur cette application de démonstration.")

elif choice == "KNN":
    st.title("K-Nearest Neighbors (KNN)")
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    afficher_resultats(y_test, y_pred)

elif choice == "LDA/QDA":
    st.title("LDA et QDA")
    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    afficher_resultats(y_test, y_pred)

elif choice == "PCA":
    st.title("Analyse en Composantes Principales (PCA)")
    model = PCA(n_components=2)
    X_pca = model.fit_transform(X)
    st.write("Réduction de dimension effectuée.")
    st.write("Variance expliquée par composante :", model.explained_variance_ratio_)

elif choice == "SVM":
    st.title("Support Vector Machines (SVM)")
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    afficher_resultats(y_test, y_pred)

elif choice == "Decision Tree":
    st.title("Arbre de Décision")
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    afficher_resultats(y_test, y_pred)
