import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# --- Configuration ---
st.set_page_config(page_title="Projet ML - S8", layout="wide")
st.title("Interface de Démonstration : Machine Learning")

@st.cache_data
def charger_donnees():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['cible'] = data.target
    return df, data.feature_names

df, features = charger_donnees()
X = df[features]
y = df['cible']

# --- Fonction Matrice de Confusion ---
def afficher_matrice(y_true, y_pred, nom_model):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    plt.title(f"Matrice de Confusion : {nom_model}")
    st.pyplot(fig)

# --- Navigation ---
menu = ["Accueil", "LDA & QDA", "PCA", "SVM", "KNN", "Arbre de Décision", "Mes Présentations"]
choix = st.sidebar.selectbox("Sélectionnez l'algorithme", menu)

# --- Logique ---
if choix == "Accueil":
    st.header("Bienvenue sur l'application")
    st.dataframe(df.head())

elif choix == "LDA & QDA":
    st.header("Analyse Discriminante (LDA & QDA)")
    lda = LinearDiscriminantAnalysis().fit(X, y)
    col1, col2 = st.columns(2)
    col1.metric("Score LDA", f"{lda.score(X, y):.2%}")
    afficher_matrice(y, lda.predict(X), "LDA")

elif choix == "PCA":
    st.header("PCA")
    n = st.slider("Nombre de composantes", 1, 4, 2)
    pca = PCA(n_components=n).fit(X)
    X_pca = pca.transform(X)
    st.write("Variance expliquée :", pca.explained_variance_ratio_)
    if n >= 2:
        fig, ax = plt.subplots()
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
        st.pyplot(fig)

elif choix == "SVM":
    st.header("SVM")
    noyau = st.selectbox("Noyau", ["linear", "rbf", "poly"])
    model = SVC(kernel=noyau).fit(X, y)
    st.write(f"Précision : {model.score(X, y):.2%}")
    afficher_matrice(y, model.predict(X), "SVM")

elif choix == "KNN":
    st.header("K-Nearest Neighbors (KNN)")
    k = st.slider("Voisins (k)", 1, 20, 5)
    knn = KNeighborsClassifier(n_neighbors=k).fit(X, y)
    st.write(f"Précision : {knn.score(X, y):.2%}")
    afficher_matrice(y, knn.predict(X), "KNN")

elif choix == "Arbre de Décision":
    st.header("Arbre de Décision")
    prof = st.slider("Profondeur", 1, 10, 3)
    clf = DecisionTreeClassifier(max_depth=prof).fit(X, y)
    st.write(f"Précision : {clf.score(X, y):.2%}")
    afficher_matrice(y, clf.predict(X), "Arbre de Décision")

elif choix == "Mes Présentations":
    st.title("Mes Supports de Cours")
    # هذا السطر يجلب جميع الملفات في المستودع تلقائياً
    fichiers = [f for f in os.listdir('.') if f.endswith(('.pdf', '.pptx', '.ipynb'))]
    for f in fichiers:
        url = f"https://github.com/wiamchentouf/S8_Machine_Learning/raw/main/{f.replace(' ', '%20')}"
        st.markdown(f"- [📥 Télécharger {f}]({url})")
