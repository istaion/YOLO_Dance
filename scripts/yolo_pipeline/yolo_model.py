import torch
import torch.nn as nn
from ultralytics import YOLO
import numpy as np
from typing import List, Tuple, Optional


class YOLOPoseClassifier(nn.Module):
    """
    Modèle de classification de poses basé sur YOLO Pose.
    Extrait les keypoints avec YOLO Pose puis classifie la pose.
    """
    
    def __init__(
        self, 
        num_classes: int = 6,
        yolo_model: str = 'yolov8n-pose.pt',
        freeze_yolo: bool = True,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.3
    ):
        """
        Args:
            num_classes: Nombre de classes de poses
            yolo_model: Chemin ou nom du modèle YOLO Pose pré-entraîné
            freeze_yolo: Si True, gèle les poids de YOLO
            hidden_dims: Dimensions des couches cachées du classificateur
            dropout: Taux de dropout
        """
        super().__init__()
        
        # Charger le modèle YOLO Pose
        self.yolo = YOLO(yolo_model)
        self.yolo_backbone = self.yolo.model
        
        # Mettre YOLO en mode évaluation pour éviter les conflits
        self.yolo_backbone.eval()
        
        # Geler les poids de YOLO si demandé
        if freeze_yolo:
            for param in self.yolo_backbone.parameters():
                param.requires_grad = False
        
        # Dimensions des keypoints (17 points x 3 coordonnées)
        self.keypoint_dim = 17 * 3
        
        # Construire le classificateur
        layers = []
        input_dim = self.keypoint_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        # Couche de sortie
        layers.append(nn.Linear(input_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        # Noms des classes
    def train(self, mode=True):
        """Override train mode to handle YOLO properly."""
        # Appeler train sur le module parent
        super().train(mode)
        # Toujours garder YOLO en eval mode
        self.yolo_backbone.eval()
        # Seul le classificateur doit être en mode train
        self.classifier.train(mode)
        return self
        
    def extract_keypoints(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extrait les keypoints des images avec YOLO Pose.
        
        Args:
            images: Tensor d'images [B, C, H, W]
            
        Returns:
            Tensor de keypoints normalisés [B, 51]
        """
        batch_size = images.shape[0]
        keypoints_batch = []
        
        # Convertir en numpy pour YOLO
        images_np = images.cpu().numpy().transpose(0, 2, 3, 1)
        
        for i in range(batch_size):
            # Prédiction YOLO
            results = self.yolo(images_np[i], verbose=False)
            
            if (
                results[0].keypoints is not None and 
                len(results[0].keypoints.xy) > 0 and 
                results[0].keypoints.xyn[0].shape[0] == 17
            ):
                kpts = results[0].keypoints.xyn[0].cpu().numpy()
                conf = results[0].keypoints.conf[0].cpu().numpy() if results[0].keypoints.conf is not None else np.ones(17)

                kpts_with_conf = np.zeros((17, 3))
                kpts_with_conf[:, :2] = kpts
                kpts_with_conf[:, 2] = conf

                keypoints_batch.append(kpts_with_conf.flatten())
            else:
                keypoints_batch.append(np.zeros(51))  # Pas de détection ou détection incomplète
        
        return torch.tensor(keypoints_batch, dtype=torch.float32).to(images.device)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass du modèle.
        
        Args:
            images: Tensor d'images [B, C, H, W]
            
        Returns:
            Logits de classification [B, num_classes]
        """
        # Extraire les keypoints
        keypoints = self.extract_keypoints(images)
        
        # Classifier la pose
        logits = self.classifier(keypoints)
        
        return logits
    
    def predict(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prédiction avec probabilités.
        
        Returns:
            (classes prédites, probabilités)
        """
        with torch.no_grad():
            logits = self.forward(images)
            probs = torch.softmax(logits, dim=1)
            classes = torch.argmax(probs, dim=1)
            
        return classes, probs


class YOLOPoseFeatureExtractor(nn.Module):
    """
    Version alternative : utilise les features du backbone YOLO
    au lieu des keypoints pour la classification.
    """
    
    def __init__(
        self,
        num_classes: int = 6,
        yolo_model: str = 'yolov8n-pose.pt',
        feature_layer: str = 'model.22',  # Couche à extraire
        freeze_layers: Optional[List[str]] = None,
        freeze_yolo: bool = False,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Charger YOLO
        self.yolo = YOLO(yolo_model)
        self.yolo_backbone = self.yolo.model
        
        # Mettre YOLO en mode évaluation
        self.yolo_backbone.eval()
        
        # Geler certaines couches
        if freeze_layers:
            for name, param in self.yolo_backbone.named_parameters():
                if any(layer in name for layer in freeze_layers):
                    param.requires_grad = False
        
        # Hook pour extraire les features
        self.features = None
        self.feature_layer = feature_layer
        self._register_hook()

        # Pooling adaptatif
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Déterminer la dimension des features (à ajuster selon le modèle)
        self.feature_dim = self._get_feature_dim()
        
        # Classificateur
        layers = []
        input_dim = self.feature_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
    
    def train(self, mode=True):
        """Override train mode to handle YOLO properly."""
        super().train(mode)
        self.yolo_backbone.eval()
        self.classifier.train(mode)
        return self
        
    def _register_hook(self):
        def hook(module, input, output):
            self.features = output[0] if isinstance(output, tuple) else output
        
        # Naviguer dans le modèle pour trouver la couche
        parts = self.feature_layer.split('.')
        layer = self.yolo_backbone
        for part in parts:
            layer = getattr(layer, part)
        layer.register_forward_hook(hook)
    
    def _get_feature_dim(self):
        """Détermine la dimension des features extraites."""
        # Faire une passe avec une image dummy
        dummy = torch.rand(1, 3, 640, 640)
        with torch.no_grad():
            _ = self.yolo(dummy, verbose=False)
            if self.features is not None:
                pooled = self.pool(self.features)
                return pooled.flatten(1).shape[1]
        return 512  # Valeur par défaut
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Extraire les features via YOLO
        _ = self.yolo(images, verbose=False)
        
        # Pooling et aplatissement
        features = self.pool(self.features)
        features = features.flatten(1)
        
        # Classification
        logits = self.classifier(features)
        
        return logits


def create_model(
    model_type: str = 'keypoint',
    num_classes: int = 6,
    **kwargs
) -> nn.Module:
    """
    Factory pour créer le modèle.
    
    Args:
        model_type: 'keypoint' ou 'feature'
        num_classes: Nombre de classes
        **kwargs: Arguments additionnels pour le modèle
        
    Returns:
        Modèle PyTorch
    """
    if model_type == 'keypoint':
        return YOLOPoseClassifier(num_classes=num_classes, **kwargs)
    elif model_type == 'feature':
        return YOLOPoseFeatureExtractor(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Type de modèle inconnu: {model_type}")


if __name__ == "__main__":
    # Test du modèle
    model = create_model('keypoint', num_classes=6)
    
    # Image de test
    dummy_input = torch.randn(2, 3, 640, 640)
    output = model(dummy_input)
    
    print(f"Forme de sortie: {output.shape}")
    print(f"Classes prédites: {torch.argmax(output, dim=1)}")
