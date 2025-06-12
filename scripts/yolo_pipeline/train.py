import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime
import json
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Ajouter le chemin pour importer yolo_model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from yolo_model import create_model


class PoseDataset(Dataset):
    """Dataset personnalisé pour les images de poses."""
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Chemin vers le dossier contenant les sous-dossiers de classes
            transform: Transformations à appliquer
        """
        self.dataset = ImageFolder(root_dir, transform=transform)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


def get_transforms(phase='train', img_size=640):
    """
    Définit les transformations pour l'entraînement et la validation.
    
    Args:
        phase: 'train' ou 'val'
        img_size: Taille des images pour YOLO
    """
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Entraîne le modèle pour une époque."""
    # Utiliser le mode train de PyTorch correctement
    if hasattr(model, 'training'):
        model.training = True
    if hasattr(model, 'classifier'):
        model.classifier.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Statistiques
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Mise à jour de la barre de progression
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Évalue le modèle sur le set de validation."""
    # Utiliser le mode eval de PyTorch correctement
    if hasattr(model, 'training'):
        model.training = False
    if hasattr(model, 'classifier'):
        model.classifier.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """Trace et sauvegarde la matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Matrice de Confusion')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_training_history(history, save_dir):
    """Sauvegarde l'historique d'entraînement."""
    # Sauvegarder en JSON
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    # Tracer les courbes
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Train')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation')
    plt.title('Loss au cours de l\'entraînement')
    plt.xlabel('Époque')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Train')
    plt.plot(epochs, history['val_acc'], 'ro-', label='Validation')
    plt.title('Accuracy au cours de l\'entraînement')
    plt.xlabel('Époque')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()


def main(args):
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation de: {device}")
    early_stop_counter = 0
    best_val_acc = 0.0
    
    # Créer le dossier de sauvegarde
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    # Sauvegarder la configuration
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Vérifier la structure du dataset
    has_train_val_split = os.path.exists(os.path.join(args.data_path, 'train')) and \
                         os.path.exists(os.path.join(args.data_path, 'val'))
    
    if has_train_val_split:
        print("Utilisation du split train/val existant")
        train_dataset = PoseDataset(
            os.path.join(args.data_path, 'train'),
            transform=get_transforms('train', args.img_size)
        )
        
        val_dataset = PoseDataset(
            os.path.join(args.data_path, 'val'),
            transform=get_transforms('val', args.img_size)
        )
        classes = train_dataset.classes
    else:
        print("Pas de split train/val trouvé. Création d'un split 80/20...")
        
        # Créer deux datasets avec les bonnes transformations dès le début
        train_transform = get_transforms('train', args.img_size)
        val_transform = get_transforms('val', args.img_size)
        
        # Dataset complet pour obtenir les indices
        temp_dataset = ImageFolder(args.data_path)
        classes = temp_dataset.classes
        
        print(f"Classes trouvées: {classes}")
        print(f"Nombre total d'images: {len(temp_dataset)}")
        
        # Créer les indices de split
        indices = list(range(len(temp_dataset)))
        train_size = int(0.8 * len(indices))
        
        # Mélanger et séparer les indices
        np.random.seed(42)
        np.random.shuffle(indices)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Créer les datasets avec les transformations appropriées
        train_dataset = torch.utils.data.Subset(
            ImageFolder(args.data_path, transform=train_transform),
            train_indices
        )
        
        val_dataset = torch.utils.data.Subset(
            ImageFolder(args.data_path, transform=val_transform),
            val_indices
        )
        
        print(f"Split créé: {len(train_dataset)} images pour l'entraînement, {len(val_dataset)} pour la validation")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Modèle
    model = create_model(
        model_type=args.model_type,
        num_classes=args.num_classes,
        yolo_model=args.yolo_model,
        freeze_yolo=args.freeze_yolo,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout
    )
    model = model.to(device)
    
    # Loss et optimiseur
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    # Scheduler
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None
    
    # Historique
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    # Boucle d'entraînement
    for epoch in range(args.epochs):
        print(f"\n===== Époque {epoch+1}/{args.epochs} =====")
        
        # Entraînement
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        
        # Mise à jour du scheduler
        if scheduler:
            scheduler.step()
        
        # Sauvegarder l'historique
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Sauvegarder le meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stop_counter = 0  # réinitialise le compteur
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': vars(args)
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"Nouveau meilleur modèle sauvegardé! (Val Acc: {val_acc:.2f}%)")
        else:
            early_stop_counter += 1
            print(f"Aucune amélioration. Compteur early stopping: {early_stop_counter}/{args.patience}")

        # Early stopping (faudra mettre encapsuler ça à un moment...)
        if args.early_stopping and early_stop_counter >= args.patience:
            print(f"\nEarly stopping déclenché après {epoch+1} époques sans amélioration.")
            # Charger le meilleur modèle pour l'évaluation finale
            checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'), map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])

            # Sauvegarder l'historique
            save_training_history(history, save_dir)

            # Évaluation finale
            print("\n===== Évaluation finale =====")
            _, _, final_preds, final_labels = validate(model, val_loader, criterion, device)
            report = classification_report(final_labels, final_preds, target_names=classes)
            print("\nRapport de classification:")
            print(report)
            with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
                f.write(report)
            plot_confusion_matrix(final_labels, final_preds, classes, 
                                os.path.join(save_dir, 'confusion_matrix.png'))
            print(f"\nEntraînement terminé par early stopping. Résultats sauvegardés dans: {save_dir}")
            return  # Quitter la fonction main()

        
        # Sauvegarder le dernier modèle
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'config': vars(args)
        }, os.path.join(save_dir, 'last_model.pth'))
    
    # Évaluation finale
    print("\n===== Évaluation finale =====")
    _, _, final_preds, final_labels = validate(model, val_loader, criterion, device)
    
    # Rapport de classification
    if not has_train_val_split:
        # Classes déjà définies dans le bloc else ci-dessus
        pass
    
    report = classification_report(final_labels, final_preds, target_names=classes)
    print("\nRapport de classification:")
    print(report)
    
    # Sauvegarder le rapport
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Matrice de confusion
    plot_confusion_matrix(final_labels, final_preds, classes, 
                         os.path.join(save_dir, 'confusion_matrix.png'))
    
    # Sauvegarder l'historique
    save_training_history(history, save_dir)
    
    print(f"\nEntraînement terminé! Résultats sauvegardés dans: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entraînement du classificateur de poses YOLO')
    
    # Données
    parser.add_argument('--data_path', type=str, default='data/dataset/images',
                        help='Chemin vers le dataset')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Dossier de sortie')
    
    # Modèle
    parser.add_argument('--model_type', type=str, default='feature',
                        choices=['keypoint', 'feature'],
                        help='Type de modèle à utiliser')
    parser.add_argument('--yolo_model', type=str, default='yolov8n-pose.pt',
                        help='Modèle YOLO à utiliser')
    parser.add_argument('--num_classes', type=int, default=6,
                        help='Nombre de classes')
    parser.add_argument('--freeze_yolo', action='store_false',
                        help='Geler les poids de YOLO')
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[256, 128],
                        help='Dimensions des couches cachées')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Taux de dropout')
    
    # Entraînement
    parser.add_argument('--epochs', type=int, default=500,
                        help='Nombre d\'époques')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Taille du batch')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'],
                        help='Optimiseur')
    parser.add_argument('--early_stopping', action='store_true',
                        help='Activer l’early stopping')
    parser.add_argument('--patience', type=int, default=20,
                        help='Nombre d’époques à attendre avant d’arrêter si pas d’amélioration')

    
    # Scheduler
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['none', 'step', 'cosine'],
                        help='Learning rate scheduler')
    parser.add_argument('--step_size', type=int, default=10,
                        help='Step size pour StepLR')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Gamma pour StepLR')
    
    # Autres
    parser.add_argument('--img_size', type=int, default=640,
                        help='Taille des images')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Nombre de workers pour le DataLoader')
    
    args = parser.parse_args()
    
    main(args)