import streamlit as st

st.title("Projet Machine Learning - S8")

menu = ["Accueil", "KNN", "LDA/QDA", "PCA", "SVM", "Decision Tree"]
choice = st.sidebar.selectbox("Choisir une méthode", menu)

if choice == "Accueil":
    st.write("Bienvenue dans l'application de démonstration ML.")

elif choice == "KNN":
    st.header("K-Nearest Neighbors (KNN)")
    st.subheader("Aspect Théorique")
    st.write("Le KNN est un algorithme non paramétrique qui classe un point selon la majorité de ses k-voisins les plus proches.")
    st.latex(r"d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}") # Exemple mathématique
    # ضعي هنا كود الـ KNN الخاص بك

elif choice == "LDA/QDA":
    st.header("LDA et QDA")
    st.write("Analyse Discriminante Linéaire et Quadratique.")
    # ضعي هنا كود الـ LDA/QDA الخاص بك

elif choice == "PCA":
    st.header("ACP (PCA)")
    st.write("Réduction de dimensionnalité par maximisation de la variance.")
    # ضعي هنا كود الـ PCA الخاص بك

elif choice == "SVM":
    st.header("Support Vector Machines (SVM)")
    st.write("Trouver l'hyperplan optimal qui sépare les classes.")
    # ضعي هنا كود الـ SVM الخاص بك

elif choice == "Decision Tree":
    st.header("Arbre de Décision")
    st.write("Modèle prédictif basé sur des règles de décision.")
    # ضعي هنا كود الـ Decision Tree الخاص بك
