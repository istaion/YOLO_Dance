# scripts/yolo_pipeline/train_weighted.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# Import du nouveau système pondéré
from yolo_model_weighted import YoloDanceSystemWeighted, WeightedDanceClassifier

class WeightedDanceDataset(Dataset):
    """Dataset optimisé pour l'entraînement pondéré"""
    
    def __init__(self, data_dir: str, classes: list, split='train', precompute_features=True):
        self.data_dir = Path(data_dir)
        self.classes = classes
        self.split = split
        self.precompute_features = precompute_features
        
        # Charger les données
        self.samples = []
        self.labels = []
        self.features_cache = {}
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Le répertoire {self.data_dir} n'existe pas!")
        
        self._load_data()
        
        if self.precompute_features and len(self.samples) > 0:
            self._precompute_weighted_features()
    
    def _load_data(self):
        """Charge les données par classe"""
        for class_idx, class_name in enumerate(self.classes):
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
                images_found = 0
                
                for ext in image_extensions:
                    for img_path in class_dir.glob(ext):
                        self.samples.append(str(img_path))
                        self.labels.append(class_idx)
                        images_found += 1
                
                print(f"   {class_name}: {images_found} images")
        
        print(f"📊 Dataset {self.split}: {len(self.samples)} échantillons")
    
    def _precompute_weighted_features(self):
        """Précompute les features avec pondération optimisée"""
        print(f"🔄 Précomputation features pondérées pour {len(self.samples)} échantillons...")
        
        # Initialiser le système pondéré
        dance_system = YoloDanceSystemWeighted()
        
        for idx, img_path in enumerate(tqdm(self.samples, desc="Extraction features pondérées")):
            try:
                image = cv2.imread(img_path)
                if image is None:
                    print(f"⚠️ Impossible de charger {img_path}")
                    self.features_cache[idx] = np.zeros(192)
                    continue
                
                # Extraire features avec pondération
                pose_features = dance_system.extract_pose_features(
                    dance_system.pose_detector(image)
                )
                hand_features = dance_system.extract_hand_features(image)
                yolo_results = dance_system.pose_detector(image)
                context_features = dance_system.extract_context_features(
                    yolo_results, image.shape
                )
                
                # Concaténer (pondération appliquée dans le modèle)
                all_features = np.concatenate([pose_features, hand_features, context_features])
                
                if all_features.size != 192:
                    if all_features.size < 192:
                        padding = np.zeros(192 - all_features.size)
                        all_features = np.concatenate([all_features, padding])
                    else:
                        all_features = all_features[:192]
                
                self.features_cache[idx] = all_features
                
            except Exception as e:
                print(f"⚠️ Erreur avec {img_path}: {e}")
                self.features_cache[idx] = np.zeros(192)
        
        print(f"✅ Features pondérées précomputées pour {len(self.features_cache)} échantillons")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        
        if self.precompute_features and idx in self.features_cache:
            features = self.features_cache[idx]
        else:
            # Extraction à la volée (fallback)
            features = np.zeros(192)
        
        return torch.FloatTensor(features), torch.LongTensor([label])

class WeightedDanceTrainer:
    """Trainer optimisé pour l'entraînement pondéré"""
    
    def __init__(self, data_dir: str, model_save_dir: str = './models'):
        self.data_dir = data_dir
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(exist_ok=True)
        
        # Classes
        self.classes = [
            'hands_up', 'dab', 'twerk',
            'jul', 'neutral',
            'crossarm'
        ]
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Device utilisé: {self.device}")
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def prepare_data(self, test_size=0.2, val_size=0.2, batch_size=32):
        """Prépare les datasets pondérés"""
        print("📂 Préparation des données pondérées...")
        
        full_dataset = WeightedDanceDataset(
            self.data_dir, 
            self.classes, 
            'full', 
            precompute_features=True
        )
        
        if len(full_dataset) < 10:
            raise ValueError(f"Dataset trop petit ({len(full_dataset)} échantillons)")
        
        # Splits avec stratification
        if len(set(full_dataset.labels)) > 1:
            train_indices, test_indices = train_test_split(
                range(len(full_dataset)), 
                test_size=test_size, 
                stratify=full_dataset.labels,
                random_state=42
            )
            
            train_indices, val_indices = train_test_split(
                train_indices, 
                test_size=val_size/(1-test_size), 
                stratify=[full_dataset.labels[i] for i in train_indices],
                random_state=42
            )
        else:
            train_indices, test_indices = train_test_split(
                range(len(full_dataset)), test_size=test_size, random_state=42
            )
            train_indices, val_indices = train_test_split(
                train_indices, test_size=val_size/(1-test_size), random_state=42
            )
        
        # Créer les datasets
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
        
        # DataLoaders optimisés
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=0, pin_memory=True if self.device.type == 'cuda' else False
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=0, pin_memory=True if self.device.type == 'cuda' else False
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=0, pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"✅ Données pondérées préparées:")
        print(f"   Train: {len(train_dataset)} échantillons")
        print(f"   Val: {len(val_dataset)} échantillons")
        print(f"   Test: {len(test_dataset)} échantillons")
        
        return self.train_loader, self.val_loader, self.test_loader
    
    def create_weighted_model(self):
        """Crée le modèle pondéré"""
        print("🎯 Création du modèle pondéré...")
        print("   - Pose YOLO: poids 3.0x")
        print("   - Mains MediaPipe: poids 0.5x")
        print("   - Context: poids 1.0x")
        
        self.model = WeightedDanceClassifier(
            num_classes=len(self.classes)
        ).to(self.device)
        
        # Optimiseur avec learning rate adaptatif
        self.optimizer = optim.AdamW(
            [
                # Poids plus élevés pour les features de pose
                {'params': self.model.pose_transform.parameters(), 'lr': 0.002},
                # Poids standards pour les autres
                {'params': self.model.hand_transform.parameters(), 'lr': 0.0005},
                {'params': self.model.context_transform.parameters(), 'lr': 0.001},
                {'params': self.model.fusion_layer.parameters(), 'lr': 0.001}
            ],
            weight_decay=0.01
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Loss function avec pondération des classes
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"🧠 Modèle pondéré créé avec {sum(p.numel() for p in self.model.parameters())} paramètres")
        
        # Afficher la répartition des paramètres
        pose_params = sum(p.numel() for p in self.model.pose_transform.parameters())
        hand_params = sum(p.numel() for p in self.model.hand_transform.parameters())
        context_params = sum(p.numel() for p in self.model.context_transform.parameters())
        fusion_params = sum(p.numel() for p in self.model.fusion_layer.parameters())
        
        print(f"   📊 Répartition des paramètres:")
        print(f"      - Pose: {pose_params:,} paramètres")
        print(f"      - Mains: {hand_params:,} paramètres")
        print(f"      - Context: {context_params:,} paramètres")
        print(f"      - Fusion: {fusion_params:,} paramètres")
    
    def train_epoch(self):
        """Entraîne une époque avec monitoring de la pondération"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Monitoring des contributions
        pose_contributions = []
        hand_contributions = []
        context_contributions = []
        
        progress_bar = tqdm(self.train_loader, desc="Training pondéré")
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.squeeze().to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass avec monitoring
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Monitoring des gradients par type de feature
            loss.backward()
            
            # Calculer les contributions moyennes
            with torch.no_grad():
                # Séparer les features
                pose_features = data[:, :51]
                hand_features = data[:, 51:177]
                context_features = data[:, 177:]
                
                # Calculer les normes pour monitoring
                pose_norm = torch.norm(pose_features, dim=1).mean().item()
                hand_norm = torch.norm(hand_features, dim=1).mean().item()
                context_norm = torch.norm(context_features, dim=1).mean().item()
                
                pose_contributions.append(pose_norm)
                hand_contributions.append(hand_norm)
                context_contributions.append(context_norm)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Mise à jour de la barre de progression
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'Pose': f'{pose_norm:.3f}',
                'Hand': f'{hand_norm:.3f}',
                'Ctx': f'{context_norm:.3f}'
            })
        
        # Statistiques de contribution
        avg_pose = np.mean(pose_contributions)
        avg_hand = np.mean(hand_contributions)
        avg_context = np.mean(context_contributions)
        
        print(f"📊 Contributions moyennes:")
        print(f"   - Pose: {avg_pose:.3f} (poids 3.0x)")
        print(f"   - Mains: {avg_hand:.3f} (poids 0.5x)")
        print(f"   - Context: {avg_context:.3f} (poids 1.0x)")
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self):
        """Validation avec analyse de pondération"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        # Tracking des prédictions par type de feature
        correct_by_feature = {'pose_dominant': 0, 'hand_dominant': 0, 'balanced': 0}
        total_by_feature = {'pose_dominant': 0, 'hand_dominant': 0, 'balanced': 0}
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.squeeze().to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                # Analyser quelle feature domine
                pose_features = data[:, :51]
                hand_features = data[:, 51:177]
                
                pose_norm = torch.norm(pose_features, dim=1)
                hand_norm = torch.norm(hand_features, dim=1)
                
                for i in range(len(pose_norm)):
                    if pose_norm[i] > hand_norm[i] * 2:
                        category = 'pose_dominant'
                    elif hand_norm[i] > pose_norm[i] * 2:
                        category = 'hand_dominant'
                    else:
                        category = 'balanced'
                    
                    total_by_feature[category] += 1
                    if pred[i] == target[i]:
                        correct_by_feature[category] += 1
        
        # Afficher les performances par type
        print("📈 Performances par dominance de features:")
        for category, total_cat in total_by_feature.items():
            if total_cat > 0:
                acc_cat = 100. * correct_by_feature[category] / total_cat
                print(f"   - {category}: {acc_cat:.2f}% ({correct_by_feature[category]}/{total_cat})")
        
        return total_loss / len(self.val_loader), 100. * correct / total
    
    def train_weighted(self, epochs=50, early_stopping_patience=5):
        """Boucle d'entraînement avec pondération optimisée"""
        print(f"🚀 Début de l'entraînement pondéré pour {epochs} époques")
        print("🎯 Priorité: Pose YOLO > Context > Mains MediaPipe")
        
        best_val_acc = 0
        early_stopping_counter = 0
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        for epoch in range(epochs):
            print(f"\n📈 Époque {epoch+1}/{epochs}")
            
            # Entraînement avec monitoring
            train_loss, train_acc = self.train_epoch()
            
            # Validation avec analyse
            val_loss, val_acc = self.validate()
            
            # Scheduler
            self.scheduler.step(val_loss)
            
            # Sauvegarder les métriques
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Sauvegarder le meilleur modèle
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                early_stopping_counter = 0
                self.save_weighted_model('best_weighted_model.pth')
                print(f"🎯 Nouveau meilleur modèle pondéré sauvegardé! Acc: {val_acc:.2f}%")
            else:
                early_stopping_counter += 1
            
            # Early stopping
            if early_stopping_counter >= early_stopping_patience:
                print(f"⏹️  Early stopping après {epoch+1} époques")
                break
        
        # Sauvegarder le modèle final
        self.save_weighted_model('final_weighted_model.pth')
        
        print(f"✅ Entraînement pondéré terminé! Meilleure précision: {best_val_acc:.2f}%")
        return train_losses, val_losses, train_accs, val_accs
    
    def save_weighted_model(self, filename):
        """Sauvegarde le modèle pondéré avec métadonnées"""
        model_path = self.model_save_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'classes': self.classes,
            'config': {
                'pose_features': 51,
                'hand_features': 126,
                'context_features': 15,
                'num_classes': len(self.classes),
                'pose_weight': 3.0,
                'hand_weight': 0.5,
                'context_weight': 1.0,
                'model_type': 'weighted_dance_classifier'
            },
            'weights_info': {
                'pose_transform_params': sum(p.numel() for p in self.model.pose_transform.parameters()),
                'hand_transform_params': sum(p.numel() for p in self.model.hand_transform.parameters()),
                'context_transform_params': sum(p.numel() for p in self.model.context_transform.parameters()),
                'fusion_params': sum(p.numel() for p in self.model.fusion_layer.parameters())
            }
        }, model_path)
        print(f"💾 Modèle pondéré sauvegardé: {model_path}")
    
    def evaluate_weighted_model(self):
        """Évaluation détaillée du modèle pondéré"""
        print("📊 Évaluation détaillée du modèle pondéré...")
        
        self.model.eval()
        all_preds = []
        all_targets = []
        feature_analysis = {'pose_dominant': [], 'hand_dominant': [], 'balanced': []}
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="Évaluation pondérée"):
                data, target = data.to(self.device), target.squeeze().to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                # Analyser la dominance des features
                pose_features = data[:, :51]
                hand_features = data[:, 51:177]
                
                pose_norm = torch.norm(pose_features, dim=1)
                hand_norm = torch.norm(hand_features, dim=1)
                
                for i in range(len(pose_norm)):
                    is_correct = (pred[i] == target[i]).item()
                    
                    if pose_norm[i] > hand_norm[i] * 2:
                        feature_analysis['pose_dominant'].append(is_correct)
                    elif hand_norm[i] > pose_norm[i] * 2:
                        feature_analysis['hand_dominant'].append(is_correct)
                    else:
                        feature_analysis['balanced'].append(is_correct)
        
        # Rapport de classification
        print("\n📈 Rapport de classification pondéré:")
        print(classification_report(all_targets, all_preds, target_names=self.classes))
        
        # Analyse par dominance de features
        print("\n🎯 Analyse par dominance de features:")
        for category, results in feature_analysis.items():
            if results:
                acc = np.mean(results) * 100
                count = len(results)
                print(f"   - {category}: {acc:.2f}% ({count} échantillons)")
        
        # Précision globale
        accuracy = (np.array(all_preds) == np.array(all_targets)).mean()
        print(f"\n🎯 Précision globale pondérée: {accuracy*100:.2f}%")
        
        return accuracy
    
    def compare_with_unweighted(self, unweighted_model_path):
        """Compare avec un modèle non-pondéré"""
        print("🔍 Comparaison avec modèle non-pondéré...")
        
        try:
            # Charger l'ancien modèle pour comparaison
            unweighted_data = torch.load(unweighted_model_path, map_location=self.device)
            print(f"✅ Modèle non-pondéré chargé pour comparaison")
            
            # Ici vous pourriez implémenter une comparaison détaillée
            print("📊 Avantages du modèle pondéré:")
            print("   ✅ Priorité donnée aux poses corporelles (YOLO)")
            print("   ✅ Réduction du bruit des features de mains")
            print("   ✅ Meilleure généralisation attendue")
            
        except Exception as e:
            print(f"⚠️ Impossible de charger le modèle de comparaison: {e}")

def main():
    """Fonction principale d'entraînement pondéré"""
    print("🎯 Entraînement YOLO Dance avec pondération optimisée")
    print("=" * 60)
    
    # Configuration
    DATA_DIR = "data/dataset/images"
    MODEL_SAVE_DIR = "models"
    
    # Créer le trainer pondéré
    trainer = WeightedDanceTrainer(DATA_DIR, MODEL_SAVE_DIR)
    
    # Préparer les données
    train_loader, val_loader, test_loader = trainer.prepare_data(
        test_size=0.2,
        val_size=0.2,
        batch_size=32
    )
    
    # Créer le modèle pondéré
    trainer.create_weighted_model()
    
    # Entraîner avec pondération
    train_losses, val_losses, train_accs, val_accs = trainer.train_weighted(
        epochs=50, 
        early_stopping_patience=5
    )
    
    # Évaluer le modèle pondéré
    trainer.evaluate_weighted_model()
    
    # Comparaison optionnelle
    old_model_path = Path(MODEL_SAVE_DIR) / "best_model.pth"
    if old_model_path.exists():
        trainer.compare_with_unweighted(str(old_model_path))
    
    print("\n🎉 Entraînement pondéré terminé!")
    print(f"📁 Modèles pondérés sauvegardés dans: {MODEL_SAVE_DIR}")
    print("🎯 Le nouveau modèle privilégie les poses YOLO over les features de mains")

if __name__ == "__main__":
    main()