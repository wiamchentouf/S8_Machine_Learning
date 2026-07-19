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

st.set_page_config(page_title="Projet ML - S8", layout="wide")

st.sidebar.title("Sommaire")
menu = ["Accueil", "KNN", "LDA", "PCA", "SVM", "Decision Tree", "Mes Présentations"]
choice = st.sidebar.selectbox("Navigation", menu)

# بيانات تجريبية
X, y = make_classification(n_samples=200, n_features=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

if choice == "Accueil":
    st.title("Bienvenue sur mon Projet ML")
    st.write("Ce projet regroupe mes travaux de S8.")

elif choice == "Mes Présentations":
    st.title("Mes Supports de cours")
    st.write("Voici mes fichiers PDF et PPT :")
    st.markdown("- [Guide ML détaillé](Guide_ML_Detaille.pdf)")
    st.markdown("- [Analyse Discriminante](Analyse_Discriminante.pptx)")
    st.markdown("- [SVM Théorie](SVM_Théorie_et_Pratique.pptx)")

# إضافة باقي الخوارزميات (KNN, LDA, PCA, SVM, Decision Tree) كما في الكود السابق...
