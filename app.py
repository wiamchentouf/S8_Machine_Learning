import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

# --- Configuration de la page ---
st.set_page_config(page_title="Interface ML", layout="wide")

# --- Chargement des données ---
@st.cache_data
def load_data():
    # تأكدي أن ملف data.csv موجود في GitHub بنفس هذا الاسم
    return pd.read_csv("data.csv")

df = load_data()
target_col = df.columns[-1]  # نفترض أن العمود الأخير هو الهدف
X = df.drop(target_col, axis=1)
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Fonction d'affichage ---
def afficher_resultats(model, nom):
    y_pred = model.predict(X_test)
    st.write(f"### Résultats pour : {nom}")
    st.write(f"**Précision du modèle :** {accuracy_score(y_test, y_pred):.2%}")
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
    plt.title("Matrice de Confusion")
    st.pyplot(fig)

# --- Sidebar ---
st.sidebar.title("Navigation")
choix = st.sidebar.selectbox("Sélectionnez l'algorithme", ["Accueil", "KNN", "SVM", "Decision Tree", "LDA", "PCA"])

# --- Logique ---
if choix == "Accueil":
    st.title("Interface de Démonstration Machine Learning")
    st.write("Bienvenue. Sélectionnez un algorithme dans le menu pour voir les analyses et les graphiques.")
    st.write("Données chargées :", df.head())

elif choix == "KNN":
    k = st.sidebar.slider("Nombre de voisins (K)", 1, 15, 3)
    model = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    afficher_resultats(model, "KNN")

elif choix == "SVM":
    model = SVC().fit(X_train, y_train)
    afficher_resultats(model, "SVM")

elif choix == "Decision Tree":
    profondeur = st.sidebar.slider("Profondeur de l'arbre", 1, 10, 3)
    model = DecisionTreeClassifier(max_depth=profondeur).fit(X_train, y_train)
    afficher_resultats(model, "Arbre de Décision")
    
    st.write("### Visualisation de l'arbre")
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_tree(model, filled=True, ax=ax)
    st.pyplot(fig)

elif choix == "LDA":
    model = LinearDiscriminantAnalysis().fit(X_train, y_train)
    afficher_resultats(model, "LDA")

elif choix == "PCA":
    st.title("Analyse en Composantes Principales (PCA)")
    n = st.sidebar.slider("Nombre de composantes", 1, min(X.shape[1], 5), 2)
    pca = PCA(n_components=n)
    X_pca = pca.fit_transform(X)
    
    fig, ax = plt.subplots()
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
    plt.title("Projection PCA 2D")
    st.pyplot(fig)
    st.write("Variance expliquée :", pca.explained_variance_ratio_)
