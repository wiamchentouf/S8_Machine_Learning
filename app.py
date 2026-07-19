import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# --- Configuration et Données ---
st.set_page_config(page_title="Application de Machine Learning", layout="wide")
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

# --- Menu ---
menu = ["Accueil", "LDA & QDA", "PCA", "SVM", "KNN", "Arbre de Décision", "Mes Présentations"]
choix = st.sidebar.selectbox("Sélectionnez l'algorithme", menu)

# --- Fonction pour afficher la Matrice de Confusion ---
def afficher_matrice(y_true, y_pred, nom_model):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    plt.title(f"Matrice de Confusion : {nom_model}")
    st.pyplot(fig)

# --- Logique ---
if choix == "Accueil":
    st.header("Bienvenue sur l'application")
    st.write("Testez différents algorithmes sur le dataset Iris.")
    st.dataframe(df.head())

elif choix == "LDA & QDA":
    st.header("Analyse Discriminante (LDA & QDA)")
    lda = LinearDiscriminantAnalysis().fit(X, y)
    qda = QuadraticDiscriminantAnalysis().fit(X, y)
    col1, col2 = st.columns(2)
    col1.metric("Score LDA", f"{lda.score(X, y):.2%}")
    col2.metric("Score QDA", f"{qda.score(X, y):.2%}")
    afficher_matrice(y, lda.predict(X), "LDA")

elif choix == "PCA":
    st.header("PCA")
    n = st.slider("Nombre de composantes", 1, 4, 2)
    pca = PCA(n_components=n).fit(X)
    st.write("Variance expliquée :", pca.explained_variance_ratio_)

elif choix == "SVM":
    st.header("SVM")
    noyau = st.selectbox("Choisir le noyau", ["linear", "rbf", "poly"])
    model = SVC(kernel=noyau).fit(X, y)
    st.write(f"Précision : {model.score(X, y):.2%}")
    afficher_matrice(y, model.predict(X), "SVM")

elif choix == "KNN":
    st.header("K-Nearest Neighbors (KNN)")
    k = st.slider("Nombre de voisins (k)", 1, 20, 5)
    knn = KNeighborsClassifier(n_neighbors=k).fit(X, y)
    st.write(f"Précision : {knn.score(X, y):.2%}")
    afficher_matrice(y, knn.predict(X), "KNN")

elif choix == "Arbre de Décision":
    st.header("Arbre de Décision")
    prof = st.slider("Profondeur maximale", 1, 10, 3)
    clf = DecisionTreeClassifier(max_depth=prof).fit(X, y)
    st.write(f"Précision : {clf.score(X, y):.2%}")
    afficher_matrice(y, clf.predict(X), "Arbre de Décision")

elif choix == "Mes Présentations":
    st.title("Mes Supports de Cours")
    st.markdown("- [Analyse Discriminante](https://github.com/wiamchentouf/S8_Machine_Learning/raw/main/Analyse%20Discriminante.pptx)")
    st.markdown("- [Guide ML Détail](https://github.com/wiamchentouf/S8_Machine_Learning/raw/main/Guide_ML_Detaille.pdf)")
    st.markdown("- [SVM Théorie](https://github.com/wiamchentouf/S8_Machine_Learning/raw/main/SVM_Théorie_et_Pratique.pptx)")
