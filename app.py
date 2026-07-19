import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# --- Configuration de la page ---
st.set_page_config(page_title="Application de Machine Learning", layout="wide")
st.title("Interface de Démonstration : Machine Learning")

# --- Chargement des données ---
@st.cache_data
def charger_donnees():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['cible'] = data.target
    return df, data.feature_names

df, features = charger_donnees()
X = df[features]
y = df['cible']

# --- Menu latéral de navigation ---
st.sidebar.title("Navigation")
menu = ["Accueil", "LDA & QDA", "PCA", "SVM", "KNN", "Arbre de Décision"]
choix = st.sidebar.selectbox("Sélectionnez l'algorithme", menu)

# --- Logique de l'application ---
if choix == "Accueil":
    st.header("Bienvenue sur l'application")
    st.write("Cette application permet de tester différents algorithmes de Machine Learning sur le dataset Iris.")
    st.subheader("Aperçu des données")
    st.dataframe(df.head())

elif choix == "LDA & QDA":
    st.header("Analyse Discriminante (LDA & QDA)")
    lda = LinearDiscriminantAnalysis().fit(X, y)
    qda = QuadraticDiscriminantAnalysis().fit(X, y)
    
    col1, col2 = st.columns(2)
    col1.metric("Score LDA", f"{lda.score(X, y):.2%}")
    col2.metric("Score QDA", f"{qda.score(X, y):.2%}")

elif choix == "PCA":
    st.header("Analyse en Composantes Principales (PCA)")
    n = st.slider("Nombre de composantes principales", 1, 4, 2)
    pca = PCA(n_components=n).fit(X)
    st.write("Ratio de variance expliquée :", pca.explained_variance_ratio_)

elif choix == "SVM":
    st.header("Support Vector Machine (SVM)")
    noyau = st.selectbox("Choisissez le noyau (kernel)", ["linear", "rbf", "poly"])
    model = SVC(kernel=noyau).fit(X, y)
    st.write(f"Précision du modèle SVM ({noyau}) :", f"{model.score(X, y):.2%}")

elif choix == "KNN":
    st.header("K-Nearest Neighbors (KNN)")
    k = st.slider("Nombre de voisins (k)", 1, 20, 5)
    knn = KNeighborsClassifier(n_neighbors=k).fit(X, y)
    st.write(f"Précision du modèle KNN (k={k}) :", f"{knn.score(X, y):.2%}")

elif choix == "Arbre de Décision":
    st.header("Arbre de Décision")
    profondeur = st.slider("Profondeur maximale de l'arbre", 1, 10, 3)
    clf = DecisionTreeClassifier(max_depth=profondeur).fit(X, y)
    st.write(f"Précision de l'arbre (profondeur {profondeur}) :", f"{clf.score(X, y):.2%}")

# --- Pied de page ---
st.sidebar.markdown("---")
st.sidebar.info("Application créée pour illustrer les concepts de Machine Learning.")
