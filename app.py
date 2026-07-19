import streamlit as st

# Configuration de la page
st.set_page_config(page_title="Projet Machine Learning S8", layout="wide")

# Titre principal
st.title("Projet Machine Learning - S8")
st.sidebar.title("Sommaire")

# Menu de navigation
menu = ["Accueil", "KNN", "LDA/QDA", "PCA", "SVM", "Decision Tree"]
choice = st.sidebar.selectbox("Sélectionnez une méthode", menu)

if choice == "Accueil":
    st.header("Bienvenue")
    st.write("Ce projet présente différentes méthodes de Machine Learning abordées durant le semestre 8.")
    st.write("Veuillez choisir une méthode dans le menu latéral pour voir la théorie, les mathématiques et les résultats.")

elif choice == "KNN":
    st.header("K-Nearest Neighbors (KNN)")
    st.subheader("Théorie")
    st.write("Le KNN est un algorithme d'apprentissage supervisé non paramétrique. Il classe un point de données en fonction de la majorité de ses k-voisins les plus proches.")
    st.subheader("Mathématiques")
    st.write("La distance euclidienne entre deux points est donnée par :")
    st.latex(r"d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}")
    st.subheader("Résultats et Analyse")
    st.write("Insérez ici vos graphiques et résultats.")

elif choice == "LDA/QDA":
    st.header("LDA et QDA")
    st.subheader("Théorie")
    st.write("L'Analyse Discriminante Linéaire (LDA) et Quadratique (QDA) cherchent à trouver les frontières de décision linéaires ou quadratiques entre les classes.")
    st.subheader("Mathématiques (LDA)")
    st.latex(r"\delta_k(x) = x^T \Sigma^{-1} \mu_k - \frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k + \log \pi_k")
    st.subheader("Résultats et Analyse")
    st.write("Visualisation des frontières de décision.")

elif choice == "PCA":
    st.header("Analyse en Composantes Principales (PCA)")
    st.subheader("Théorie")
    st.write("La PCA est une technique de réduction de dimensionnalité qui transforme des variables corrélées en composantes principales non corrélées.")
    st.subheader("Mathématiques")
    st.write("Maximisation de la variance :")
    st.latex(r"Var(X) = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})^2")
    st.subheader("Résultats et Analyse")
    st.write("Projection des données sur les axes principaux.")

elif choice == "SVM":
    st.header("Support Vector Machines (SVM)")
    st.subheader("Théorie")
    st.write("Le SVM cherche l'hyperplan optimal qui sépare les classes avec la plus grande marge possible.")
    st.subheader("Mathématiques")
    st.write("Optimisation de la marge :")
    st.latex(r"\min_{w, b} \frac{1}{2} \|w\|^2 \quad \text{sous contrainte} \quad y_i(w^T x_i + b) \geq 1")
    st.subheader("Résultats et Analyse")
    st.write("Affichage des vecteurs de support.")

elif choice == "Decision Tree":
    st.header("Arbre de Décision")
    st.subheader("Théorie")
    st.write("L'arbre de décision est un modèle prédictif utilisant une structure en arbre pour classer les données.")
    st.subheader("Mathématiques (Impureté de Gini)")
    st.latex(r"Gini = 1 - \sum_{i=1}^C (p_i)^2")
    st.subheader("Résultats et Analyse")
    st.write("Visualisation de l'arbre de décision.")
