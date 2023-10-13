import pickle
import pandas as pd
import numpy as np

# Charger le modèle
model = pickle.load(open("model.pkl", "rb"))

# Supposez que 'feature_names' est une liste contenant les noms de colonnes de votre modèle
feature_names = model.get_booster().feature_names




# Générer des données de test où toutes les valeurs sont 1
num_samples = 10  # Nombre d'échantillons à générer
num_features = 261  # Nombre de variables dans votre nouveau modèle
X_test = pd.DataFrame(np.ones((num_samples, num_features)), columns=feature_names)

# Prédire avec le modèle
y_pred = model.predict(X_test)


# Vérifier que le résultat est comme prévu
assert y_pred.shape[0] == X_test.shape[0]  # Vérifier que le nombre de prédictions est correct
assert (y_pred >= 0).all()  # Vérifier que toutes les prédictions sont supérieures ou égales à 0
assert (y_pred <= 1).all()  # Vérifier que toutes les prédictions sont inférieures ou égales à 1
