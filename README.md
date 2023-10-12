# Projet 7: ["Scoring Modèle"]

Ce projet vise à développer un modèle de scoring robuste et de le déployer via une API, tout en assurant une visualisation via un tableau de bord.

## Structure du Répertoire

- **Assoli_Ayad_projet7_080203.ipynb**: Le notebook principal contenant tout le travail de prétraitement, d'analyse, de modélisation et d'évaluation.
  
- **dashboard_new.py**: Script Python pour la création et la gestion du tableau de bord interactif.

- **mlflow_model**: Dossier associé à MLFlow pour le tracking d’expérimentations et le stockage centralisé des modèles.

- **pipeline_model.joblib**: Fichier contenant le modèle entraîné sous forme de pipeline, prêt à être utilisé pour la prédiction.

- **projet7Notes.pdf**: Notes et références associées au projet.

- **projet7_support presentation.pdf**: Présentation détaillant le travail réalisé pour ce projet.

- **report.html**: Rapport HTML d'analyse de data drift réalisé à partir d'evidently.

- **templates**: Dossier contenant les templates associés à l'interface utilisateur du tableau de bord.

## Packages Utilisés

- pandas
- numpy
- scikit-learn
- evidently
- mlflow
- flask

## Installation et Exécution

1. Clonez ce répertoire sur votre machine locale.
2. Assurez-vous d'avoir tous les packages nécessaires installés (voir dossier mlflow_model). Vous pouvez les installer en utilisant `pip install -r requirements.txt` .
3. Exécutez `dashboard_new.py` pour lancer le tableau de bord interactif.


## Auteur

Assoli Ayad


