import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.layers import TFSMLayer

# Chargement du modèle
tfsm_layer = TFSMLayer("converted_model", call_endpoint="serving_default")

# Définition d’une entrée correcte : vecteur de 14739 float32
inputs = Input(shape=(14739,), dtype=tf.float32)
outputs = tfsm_layer(inputs)
model = Model(inputs, outputs)

# Génération d’un vecteur de test aléatoire (ou charger depuis tes features si disponibles)
X_test = np.random.rand(1, 14739).astype("float32")  # ← à remplacer par tes vraies données
y_test = np.random.rand(1, 14739).astype("float32")
# Prédiction
pred = model.predict(X_test)
print("Prediction:", pred)

# Suppose que X_test et y_test sont déjà chargés et correctement formatés
y_pred = model.predict(X_test)

# Si y_pred contient des probabilités, on prend l'argmax pour avoir les classes
y_pred_array = y_pred['dense_Dense4']  # extrait les probabilités
y_pred_classes = np.argmax(y_pred_array, axis=1)
y_true_classes = np.argmax(y_test, axis=1)  # si y_test est one-hot
# OU simplement y_test si ce sont des entiers

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Matrice de confusion
cm = confusion_matrix(y_true_classes, y_pred_classes)

# Affichage avec Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de Confusion')
plt.xlabel('Prédit')
plt.ylabel('Vrai')
plt.show()

# Rapport de classification
print("Classification Report:")
print(classification_report(y_true_classes, y_pred_classes))
