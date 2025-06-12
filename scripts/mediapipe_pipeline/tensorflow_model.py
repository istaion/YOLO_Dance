"""
Modèle TensorFlow multi-tâches pour classification de poses et régression de position du visage
"""

# FORCER CPU dès le début pour éviter les problèmes CUDA
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Désactiver GPU complètement
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Réduire les logs TensorFlow

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, metrics
import numpy as np
import json
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import joblib

print("💻 Utilisation du CPU forcée pour éviter les problèmes CUDA")
print(f"🔧 TensorFlow version: {tf.__version__}")
print(f"🔧 Keras version: {keras.__version__}")

class PoseClassificationModel:
    """
    Modèle TensorFlow pour classification de poses avec détection de position du visage
    Architecture multi-tâches: classification + régression
    """
    
    def __init__(self, num_classes: int = 9, input_dim: int = 110):
        """
        Initialise le modèle
        
        Args:
            num_classes: Nombre de classes de poses
            input_dim: Dimension des features d'entrée
        """
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.model = None
        self.label_encoder = LabelEncoder()
        self.history = None
        
        # Classes de poses
        self.classes = ['hands_up', 'dab', 'twerk', 'mic_drop', 'middle_finger', 
                       'peace', 'heart', 'jule', 'neutral']
        
        # Configuration d'entraînement
        self.config = {
            'batch_size': 32,
            'epochs': 100,
            'validation_split': 0.2,
            'learning_rate': 0.001,
            'dropout_rate': 0.3,
            'l2_reg': 0.01
        }
    
    def build_model(self) -> keras.Model:
        """
        Construit l'architecture multi-tâches du modèle
        
        Returns:
            Modèle TensorFlow compilé
        """
        
        # Input layer
        inputs = layers.Input(shape=(self.input_dim,), name='features_input')
        
        # Couches partagées (feature extraction)
        x = layers.Dense(256, activation='relu', 
                        kernel_regularizer=keras.regularizers.l2(self.config['l2_reg']),
                        name='shared_dense_1')(inputs)
        x = layers.BatchNormalization(name='shared_bn_1')(x)
        x = layers.Dropout(self.config['dropout_rate'], name='shared_dropout_1')(x)
        
        x = layers.Dense(128, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(self.config['l2_reg']),
                        name='shared_dense_2')(x)
        x = layers.BatchNormalization(name='shared_bn_2')(x)
        x = layers.Dropout(self.config['dropout_rate'], name='shared_dropout_2')(x)
        
        x = layers.Dense(64, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(self.config['l2_reg']),
                        name='shared_dense_3')(x)
        x = layers.BatchNormalization(name='shared_bn_3')(x)
        
        # Branche de classification des poses
        classification_branch = layers.Dense(32, activation='relu', 
                                           name='classification_dense')(x)
        classification_branch = layers.Dropout(self.config['dropout_rate'], 
                                             name='classification_dropout')(classification_branch)
        
        # Sortie classification (softmax pour probabilités)
        pose_classification = layers.Dense(self.num_classes, 
                                         activation='softmax', 
                                         name='pose_classification')(classification_branch)
        
        # Branche de régression pour position du visage
        regression_branch = layers.Dense(16, activation='relu', 
                                       name='regression_dense')(x)
        regression_branch = layers.Dropout(self.config['dropout_rate'] * 0.5, 
                                         name='regression_dropout')(regression_branch)
        
        # Sortie régression (position x, y du visage + score de confiance)
        face_position = layers.Dense(3, activation='sigmoid', 
                                   name='face_position')(regression_branch)
        
        # Modèle final avec sorties multiples
        model = keras.Model(inputs=inputs, 
                          outputs=[pose_classification, face_position],
                          name='PoseClassificationModel')
        
        # Compilation avec losses et métriques multiples
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss={
                'pose_classification': 'categorical_crossentropy',
                'face_position': 'mse'  # Mean Squared Error pour régression
            },
            loss_weights={
                'pose_classification': 1.0,  # Poids principal sur classification
                'face_position': 0.3         # Poids plus faible sur régression
            },
            metrics={
                'pose_classification': ['accuracy', metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')],
                'face_position': ['mae']  # Mean Absolute Error
            }
        )
        
        self.model = model
        return model
    
    def prepare_data(self, features_list: List) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prépare les données pour l'entraînement
        
        Args:
            features_list: Liste des features engineerées
            
        Returns:
            X: Features d'entrée
            y_classification: Labels encodés pour classification
            y_regression: Positions du visage pour régression
        """
        
        print("📊 Préparation des données...")
        
        # Extraction des features et labels
        X = np.array([f.combined_features for f in features_list])
        class_labels = [f.class_label for f in features_list]
        face_positions = np.array([list(f.face_center) + [f.confidence_score] 
                                 for f in features_list])
        
        # Encodage des labels de classification
        y_classification_encoded = self.label_encoder.fit_transform(class_labels)
        y_classification = keras.utils.to_categorical(y_classification_encoded, 
                                                    num_classes=self.num_classes)
        
        print(f"✅ Données préparées:")
        print(f"  - Shape des features: {X.shape}")
        print(f"  - Shape classification: {y_classification.shape}")
        print(f"  - Shape régression: {face_positions.shape}")
        print(f"  - Classes: {list(self.label_encoder.classes_)}")
        
        # Statistiques par classe
        unique, counts = np.unique(class_labels, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"  - {cls}: {count} échantillons")
        
        return X, y_classification, face_positions
    
    def create_data_augmentation(self, X: np.ndarray, y_class: np.ndarray, 
                               y_face: np.ndarray, augment_factor: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Augmente les données avec du bruit léger et des variations
        
        Args:
            X, y_class, y_face: Données originales
            augment_factor: Facteur de multiplication des données
            
        Returns:
            Données augmentées
        """
        
        print(f"🔄 Augmentation des données (facteur {augment_factor})...")
        
        augmented_X = [X]
        augmented_y_class = [y_class]
        augmented_y_face = [y_face]
        
        for i in range(augment_factor - 1):
            # Ajout de bruit gaussien léger
            noise_std = 0.05  # 5% de bruit
            noisy_X = X + np.random.normal(0, noise_std, X.shape)
            
            # Légère variation sur la position du visage
            face_noise = np.random.normal(0, 0.02, y_face.shape)  # 2% de variation
            noisy_face = np.clip(y_face + face_noise, 0, 1)  # Garder dans [0,1]
            
            augmented_X.append(noisy_X)
            augmented_y_class.append(y_class)  # Labels inchangés
            augmented_y_face.append(noisy_face)
        
        # Concaténer toutes les données
        final_X = np.concatenate(augmented_X, axis=0)
        final_y_class = np.concatenate(augmented_y_class, axis=0)
        final_y_face = np.concatenate(augmented_y_face, axis=0)
        
        print(f"✅ Augmentation terminée: {final_X.shape[0]} échantillons totaux")
        
        return final_X, final_y_class, final_y_face
    
    def train_model(self, features_list: List, use_augmentation: bool = True,
                   save_path: str = "../../models") -> Dict:
        """
        Entraîne le modèle
        
        Args:
            features_list: Liste des features engineerées
            use_augmentation: Utiliser l'augmentation de données
            save_path: Chemin de sauvegarde du modèle
            
        Returns:
            Historique d'entraînement
        """
        
        # Préparer les données
        X, y_class, y_face = self.prepare_data(features_list)
        
        # Augmentation de données optionnelle
        if use_augmentation:
            X, y_class, y_face = self.create_data_augmentation(X, y_class, y_face)
        
        # Split train/validation/test
        X_temp, X_test, y_class_temp, y_class_test, y_face_temp, y_face_test = train_test_split(
            X, y_class, y_face, test_size=0.15, stratify=np.argmax(y_class, axis=1), 
            random_state=42
        )
        
        X_train, X_val, y_class_train, y_class_val, y_face_train, y_face_val = train_test_split(
            X_temp, y_class_temp, y_face_temp, test_size=0.2, 
            stratify=np.argmax(y_class_temp, axis=1), random_state=42
        )
        
        print(f"📈 Splits d'entraînement:")
        print(f"  - Train: {X_train.shape[0]} échantillons")
        print(f"  - Validation: {X_val.shape[0]} échantillons")
        print(f"  - Test: {X_test.shape[0]} échantillons")
        
        # Construire le modèle
        self.build_model()
        
        print(f"\n🏗️  Architecture du modèle:")
        self.model.summary()
        
        # Callbacks pour CPU (simplifiés)
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_pose_classification_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1,
                mode='max'  # Maximiser l'accuracy
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=1,
                mode='min'  # Minimiser la loss
            )
        ]
        
        # Sauvegarder seulement à la fin pour éviter les problèmes de device
        os.makedirs(save_path, exist_ok=True)
        
        # Entraînement sur CPU
        print(f"\n🚀 Début de l'entraînement sur CPU...")
        print(f"⏱️  Cela sera plus lent qu'avec GPU mais plus stable")
        
        self.history = self.model.fit(
            X_train,
            {'pose_classification': y_class_train, 'face_position': y_face_train},
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_data=(X_val, {'pose_classification': y_class_val, 'face_position': y_face_val}),
            callbacks=callbacks,
            verbose=1
        )
        
        # Évaluation sur le test set
        print(f"\n📊 Évaluation finale sur le test set:")
        test_results = self.model.evaluate(
            X_test, 
            {'pose_classification': y_class_test, 'face_position': y_face_test},
            verbose=0
        )
        
        # Affichage des résultats avec noms des métriques
        metric_names = self.model.metrics_names
        print(f"Métriques de test:")
        for name, value in zip(metric_names, test_results):
            print(f"  - {name}: {value:.4f}")
        
        # Trouver l'index de l'accuracy
        accuracy_idx = None
        mae_idx = None
        for i, name in enumerate(metric_names):
            if 'pose_classification_accuracy' in name:
                accuracy_idx = i
            elif 'face_position_mae' in name:
                mae_idx = i
        
        # Prédictions pour métriques détaillées
        predictions = self.model.predict(X_test)
        y_pred_class = np.argmax(predictions[0], axis=1)
        y_true_class = np.argmax(y_class_test, axis=1)
        
        # Rapport de classification
        print(f"\n📋 Rapport de classification:")
        class_names = [self.label_encoder.classes_[i] for i in range(len(self.label_encoder.classes_))]
        print(classification_report(y_true_class, y_pred_class, target_names=class_names))
        
        # Sauvegarder le modèle et les métadonnées
        os.makedirs(save_path, exist_ok=True)
        
        # Modèle final
        self.model.save(os.path.join(save_path, 'pose_model_final.keras'))
        
        # Label encoder
        joblib.dump(self.label_encoder, os.path.join(save_path, 'label_encoder.joblib'))
        
        # Configuration et métadonnées
        model_info = {
            'classes': self.classes,
            'num_classes': self.num_classes,
            'input_dim': self.input_dim,
            'config': self.config,
            'test_accuracy': float(test_results[accuracy_idx]) if accuracy_idx else 0.0,
            'test_mae_face': float(test_results[mae_idx]) if mae_idx else 0.0,
            'all_test_metrics': {name: float(value) for name, value in zip(metric_names, test_results)}
        }
        
        with open(os.path.join(save_path, 'model_info.json'), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"\n💾 Modèle sauvegardé dans: {save_path}")
        if accuracy_idx:
            print(f"✅ Précision test: {test_results[accuracy_idx]:.3f}")
        if mae_idx:
            print(f"✅ MAE position visage: {test_results[mae_idx]:.4f}")
        
        # Sauvegarder les données de test pour analyse ultérieure
        test_data = {
            'X_test': X_test,
            'y_class_test': y_class_test,
            'y_face_test': y_face_test,
            'predictions_class': predictions[0],
            'predictions_face': predictions[1]
        }
        
        return {
            'history': self.history.history,
            'test_results': test_results,
            'test_data': test_data,
            'model_info': model_info
        }
    
    def plot_training_history(self, save_path: str = "../../models"):
        """Visualise l'historique d'entraînement"""
        
        if self.history is None:
            print("❌ Pas d'historique d'entraînement disponible")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy de classification
        axes[0, 0].plot(self.history.history['pose_classification_accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_pose_classification_accuracy'], label='Validation')
        axes[0, 0].set_title('Précision de Classification')
        axes[0, 0].set_xlabel('Époque')
        axes[0, 0].set_ylabel('Précision')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss totale
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Loss Totale')
        axes[0, 1].set_xlabel('Époque')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Loss de classification
        axes[1, 0].plot(self.history.history['pose_classification_loss'], label='Train')
        axes[1, 0].plot(self.history.history['val_pose_classification_loss'], label='Validation')
        axes[1, 0].set_title('Loss Classification')
        axes[1, 0].set_xlabel('Époque')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # MAE régression visage
        axes[1, 1].plot(self.history.history['face_position_mae'], label='Train')
        axes[1, 1].plot(self.history.history['val_face_position_mae'], label='Validation')
        axes[1, 1].set_title('MAE Position Visage')
        axes[1, 1].set_xlabel('Époque')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'training_history.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Graphiques sauvegardés: {os.path.join(save_path, 'training_history.png')}")
    
    def load_model(self, model_path: str) -> bool:
        """
        Charge un modèle pré-entraîné
        
        Args:
            model_path: Chemin vers le dossier du modèle
            
        Returns:
            True si chargement réussi
        """
        try:
            # Charger le modèle
            self.model = keras.models.load_model(os.path.join(model_path, 'pose_model_final.keras'))
            
            # Charger le label encoder
            self.label_encoder = joblib.load(os.path.join(model_path, 'label_encoder.joblib'))
            
            # Charger les métadonnées
            with open(os.path.join(model_path, 'model_info.json'), 'r') as f:
                model_info = json.load(f)
            
            self.num_classes = model_info['num_classes']
            self.input_dim = model_info['input_dim']
            self.classes = model_info['classes']
            
            print(f"✅ Modèle chargé depuis: {model_path}")
            print(f"  - Précision test: {model_info.get('test_accuracy', 'N/A')}")
            print(f"  - Classes: {self.classes}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            return False
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Effectue une prédiction
        
        Args:
            features: Features d'entrée (shape: [batch_size, input_dim] ou [input_dim])
            
        Returns:
            (probabilities_classes, position_visage)
        """
        if self.model is None:
            raise ValueError("Modèle non chargé. Utilisez load_model() ou train_model()")
        
        # Assurer la bonne forme
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Prédiction
        predictions = self.model.predict(features, verbose=0)
        pose_probs = predictions[0]
        face_position = predictions[1]
        
        return pose_probs, face_position


# Exemple d'utilisation
if __name__ == "__main__":
    
    # Chemins
    FEATURES_PATH = "../../data/features/engineered_features.json"
    MODEL_SAVE_PATH = "../../models"
    
    # Charger les features
    print("📂 Chargement des features...")
    
    # Simuler le chargement des features (à adapter selon votre structure)
    # En réalité, vous devriez charger depuis feature_engineering.py
    with open(FEATURES_PATH, 'r') as f:
        features_data = json.load(f)
    
    # Convertir en liste d'objets EngineeredFeatures (simulation)
    from feature_engineering import EngineeredFeatures
    features_list = []
    
    for feature_dict in features_data:
        features = EngineeredFeatures(
            body_features=np.array(feature_dict['body_features']),
            hand_features=np.array(feature_dict['hand_features']),
            combined_features=np.array(feature_dict['combined_features']),
            confidence_score=feature_dict['confidence_score'],
            face_center=tuple(feature_dict['face_center']),
            person_id=feature_dict['person_id'],
            class_label=feature_dict['class_label'],
            image_path=feature_dict['image_path']
        )
        features_list.append(features)
    
    print(f"✅ {len(features_list)} échantillons chargés")
    
    # Initialiser le modèle
    model = PoseClassificationModel(
        num_classes=9,
        input_dim=110  # Dimension des features combinées
    )
    
    # Entraîner le modèle sur CPU
    print("\n🚀 Entraînement du modèle...")
    results = model.train_model(features_list, use_augmentation=True, save_path=MODEL_SAVE_PATH)
    
    # Visualiser l'entraînement
    model.plot_training_history(MODEL_SAVE_PATH)
    
    print("\n✅ Entraînement terminé!")
    print(f"🎯 Modèle sauvegardé dans: {MODEL_SAVE_PATH}")
    
    # Test de chargement
    print("\n🧪 Test de rechargement du modèle...")
    test_model = PoseClassificationModel()
    if test_model.load_model(MODEL_SAVE_PATH):
        print("✅ Rechargement réussi!")
        
        # Test de prédiction
        test_features = features_list[0].combined_features
        pose_probs, face_pos = test_model.predict(test_features)
        
        predicted_class = test_model.label_encoder.classes_[np.argmax(pose_probs[0])]
        print(f"🎯 Test de prédiction:")
        print(f"  - Classe prédite: {predicted_class}")
        print(f"  - Confiance: {np.max(pose_probs[0]):.3f}")
        print(f"  - Position visage: ({face_pos[0][0]:.3f}, {face_pos[0][1]:.3f})")
    
    print("\n🎉 Pipeline TensorFlow prêt !")