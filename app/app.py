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
from sklearn import tree

# --- Configuration de la page ---
st.set_page_config(page_title="Projet ML - S8", layout="wide")

# --- Création de données ---
@st.cache_data
def get_data():
    X, y = make_classification(n_samples=300, n_features=5, n_classes=2, random_state=42)
    df = pd.DataFrame(X, columns=[f'Var_{i}' for i in range(5)])
    df['target'] = y
    return df, X, y

df, X, y = get_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Menu latéral ---
st.sidebar.title("Sommaire")
menu = ["Accueil", "KNN", "LDA", "PCA", "SVM", "Decision Tree"]
choice = st.sidebar.selectbox("Choisissez une méthode", menu)

# --- Fonction Résultats ---
def afficher_resultats(model, X_test, y_test):
    y_pred = model.predict(X_test)
    st.write(f"### Précision (Accuracy) : {accuracy_score(y_test, y_pred):.2%}")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    st.pyplot(fig)

# --- Logique des algorithmes ---
if choice == "Accueil":
    st.title("Projet Machine Learning")
    st.write("Bienvenue, sélectionnez un algorithme dans le menu.")

elif choice == "KNN":
    st.title("K-Nearest Neighbors")
    model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    afficher_resultats(model, X_test, y_test)

elif choice == "LDA":
    st.title("Linear Discriminant Analysis")
    model = LinearDiscriminantAnalysis().fit(X_train, y_train)
    afficher_resultats(model, X_test, y_test)

elif choice == "PCA":
    st.title("PCA (Réduction de dimension)")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    fig, ax = plt.subplots()
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm')
    st.pyplot(fig)

elif choice == "SVM":
    st.title("Support Vector Machines")
    model = SVC(kernel='linear').fit(X_train, y_train)
    afficher_resultats(model, X_test, y_test)

elif choice == "Decision Tree":
    st.title("Arbre de Décision")
    model = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)
    afficher_resultats(model, X_test, y_test)
    st.write("Visualisation de l'arbre :")
    fig, ax = plt.subplots(figsize=(10, 5))
    tree.plot_tree(model, filled=True, ax=ax)
    st.pyplot(fig)
