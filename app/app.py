import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# --- Configuration de la page ---
st.set_page_config(page_title="Projet ML - S8", layout="wide")

# --- Menu latéral ---
st.sidebar.title("Sommaire")
menu = ["Accueil", "KNN", "LDA/QDA", "PCA", "SVM", "Decision Tree"]
choice = st.sidebar.selectbox("Choisissez une méthode", menu)

# --- Fonction pour afficher les résultats ---
def afficher_resultats(y_test, y_pred):
    """
    Cette fonction calcule et affiche l'accuracy et la matrice de confusion.
    """
    # Calcul de l'accuracy
    acc = accuracy_score(y_test, y_pred)
    st.write(f"### Précision (Accuracy) : {acc:.2%}")
    
    # Création de la heatmap pour la matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matrice de Confusion")
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    
    # Affichage du graphique dans Streamlit
    st.pyplot(fig)

# --- Contenu principal ---
if choice == "Accueil":
    st.title("Projet Machine Learning - S8")
    st.write("Bienvenue sur cette application interactive de démonstration.")
    st.write("Veuillez sélectionner une méthode dans le menu à gauche.")

elif choice == "KNN":
    st.title("K-Nearest Neighbors (KNN)")
    st.subheader("Aspect Théorique")
    st.write("Algorithme non paramétrique basé sur la proximité des voisins.")
    st.subheader("Mathématiques")
    st.latex(r"d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}")
    st.subheader("Résultats")
    # Insérez ici : y_pred = model_knn.predict(X_test)
    # afficher_resultats(y_test, y_pred)
    st.info("Code KNN à intégrer ici.")

elif choice == "LDA/QDA":
    st.title("LDA et QDA")
    st.subheader("Aspect Théorique")
    st.write("Analyse Discriminante Linéaire et Quadratique.")
    st.subheader("Mathématiques")
    st.latex(r"\delta_k(x) = x^T \Sigma^{-1} \mu_k - \frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k + \log \pi_k")
    st.subheader("Résultats")
    st.info("Code LDA/QDA à intégrer ici.")

elif choice == "PCA":
    st.title("Analyse en Composantes Principales (PCA)")
    st.subheader("Aspect Théorique")
    st.write("Technique de réduction de dimensionnalité.")
    st.subheader("Mathématiques")
    st.latex(r"Var(X) = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})^2")
    st.subheader("Résultats")
    st.info("Graphique de variance expliquée à afficher.")

elif choice == "SVM":
    st.title("Support Vector Machines (SVM)")
    st.subheader("Aspect Théorique")
    st.write("Recherche de l'hyperplan optimal séparateur.")
    st.subheader("Mathématiques")
    st.latex(r"\min_{w, b} \frac{1}{2} \|w\|^2 \quad \text{sous contrainte} \quad y_i(w^T x_i + b) \geq 1")
    st.subheader("Résultats")
    # Exemple : afficher_resultats(y_test, y_pred_svm)
    st.info("Code SVM à intégrer ici.")

elif choice == "Decision Tree":
    st.title("Arbre de Décision")
    st.subheader("Aspect Théorique")
    st.write("Structure en arbre pour la classification.")
    st.subheader("Mathématiques (Gini)")
    st.latex(r"Gini = 1 - \sum_{i=1}^C (p_i)^2")
    st.subheader("Résultats")
    st.info("Visualisation de l'arbre à intégrer.")
