#  Amazon Review Classification (NLP)

Ce projet a pour but de prédire la **note attribuée par un utilisateur** à un produit Amazon à partir de l’analyse du texte de son avis.  
Il utilise des techniques de traitement du langage naturel (NLP) et de machine learning pour classifier les avis en plusieurs catégories de score.

---

##  Objectifs

- Extraire et traiter les textes d’avis clients
- Appliquer des techniques de vectorisation (TF-IDF, n-grammes)
- Entraîner plusieurs modèles de classification :
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
  - [Optionnel] XGBoost, Naive Bayes
- Classifier les notes selon deux approches :
  - **Binaire** : score 5 vs autres
  - **Tri-classe** : Négatif (1-2), Neutre (3), Positif (4-5)

---

##  Pipeline de traitement

1. Nettoyage du texte (lowercase, ponctuation, stopwords, lemmatisation)
2. Vectorisation avec **TF-IDF** (uni- et bi-grammes)
3. Modélisation avec plusieurs classifieurs
4. Évaluation via matrices de confusion et métriques (précision, rappel, F1)

---

##  Résultats – Classification tri-classe (SVM)

| Classe   | Précision | Rappel | F1-score |
|----------|-----------|--------|----------|
| Négatif  | 0.72      | 0.67   | 0.70     |
| Neutre   | 0.56      | 0.09   | 0.16     |
| Positif  | 0.89      | 0.97   | 0.93     |
| **Accuracy globale** | –     | –       | **0.86** |

---

##  Visualisations

![Confusion Matrix SVM](img/confusion_matrix_svm.png)
