"""
Script pour télécharger et préparer le dataset COCO Keypoints
"""

import os
import urllib.request
import zipfile
import json
from tqdm import tqdm
import shutil

class COCODownloader:
    def __init__(self, base_dir='./coco_dataset'):
        self.base_dir = base_dir
        self.urls = {
            # Images (attention: fichiers volumineux!)
            'train_images': 'http://images.cocodataset.org/zips/train2017.zip',  # ~18GB
            'val_images': 'http://images.cocodataset.org/zips/val2017.zip',      # ~1GB
            
            # Annotations (léger)
            'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'  # ~240MB
        }
        
        self.setup_directories()
    
    def setup_directories(self):
        """Crée la structure de dossiers nécessaire"""
        directories = [
            self.base_dir,
            f"{self.base_dir}/images",
            f"{self.base_dir}/annotations",
            f"{self.base_dir}/processed",
            f"{self.base_dir}/processed/images/train",
            f"{self.base_dir}/processed/images/val", 
            f"{self.base_dir}/processed/labels/train",
            f"{self.base_dir}/processed/labels/val"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        print("✅ Structure de dossiers créée")
    
    def download_with_progress(self, url, filename):
        """Télécharge un fichier avec barre de progression"""
        def progress_hook(block_num, block_size, total_size):
            if hasattr(progress_hook, 'pbar'):
                progress_hook.pbar.update(block_size)
            else:
                progress_hook.pbar = tqdm(total=total_size, unit='B', unit_scale=True)
                progress_hook.pbar.set_description(f"Téléchargement {os.path.basename(filename)}")
        
        try:
            urllib.request.urlretrieve(url, filename, progress_hook)
            if hasattr(progress_hook, 'pbar'):
                progress_hook.pbar.close()
            print(f"✅ Téléchargé: {filename}")
            return True
        except Exception as e:
            print(f"❌ Erreur téléchargement {filename}: {e}")
            return False
    
    def download_annotations_only(self):
        """Télécharge SEULEMENT les annotations (pour commencer rapidement)"""
        print("📥 Téléchargement des annotations COCO...")
        
        annotations_file = f"{self.base_dir}/annotations_trainval2017.zip"
        
        if not os.path.exists(annotations_file):
            success = self.download_with_progress(
                self.urls['annotations'], 
                annotations_file
            )
            if not success:
                return False
        
        # Extraction
        print("📦 Extraction des annotations...")
        with zipfile.ZipFile(annotations_file, 'r') as zip_ref:
            zip_ref.extractall(self.base_dir)
        
        print("✅ Annotations COCO prêtes!")
        return True
    
    def download_subset_images(self, max_train=1000, max_val=200):
        """
        Télécharge un subset d'images pour POC rapide
        (évite de télécharger 18GB)
        """
        print(f"🎯 Téléchargement d'un subset: {max_train} train + {max_val} val images")
        
        # Charger les annotations pour connaître les images nécessaires
        train_ann_file = f"{self.base_dir}/annotations/person_keypoints_train2017.json"
        val_ann_file = f"{self.base_dir}/annotations/person_keypoints_val2017.json"
        
        if not os.path.exists(train_ann_file):
            print("❌ Annotations non trouvées. Lancez d'abord download_annotations_only()")
            return False
        
        # Lire les annotations et extraire les IDs d'images nécessaires
        with open(train_ann_file, 'r') as f:
            train_data = json.load(f)
        
        with open(val_ann_file, 'r') as f:
            val_data = json.load(f)
        
        # Obtenir les images avec keypoints
        train_imgs_with_keypoints = self.get_images_with_keypoints(train_data, max_train)
        val_imgs_with_keypoints = self.get_images_with_keypoints(val_data, max_val)
        
        print(f"📊 Images sélectionnées: {len(train_imgs_with_keypoints)} train, {len(val_imgs_with_keypoints)} val")
        
        # Télécharger les images individuellement
        self.download_specific_images(train_imgs_with_keypoints, 'train2017')
        self.download_specific_images(val_imgs_with_keypoints, 'val2017')
        
        return True
    
    def get_images_with_keypoints(self, coco_data, max_images):
        """Extrait les images qui ont des annotations keypoints"""
        images_with_kpts = []
        img_id_to_info = {img['id']: img for img in coco_data['images']}
        
        # Grouper annotations par image
        img_annotations = {}
        for ann in coco_data['annotations']:
            if 'keypoints' in ann and ann['num_keypoints'] > 5:  # Au moins 5 keypoints visibles
                img_id = ann['image_id']
                if img_id not in img_annotations:
                    img_annotations[img_id] = []
                img_annotations[img_id].append(ann)
        
        # Sélectionner les meilleures images
        for img_id in list(img_annotations.keys())[:max_images]:
            if img_id in img_id_to_info:
                img_info = img_id_to_info[img_id]
                images_with_kpts.append({
                    'id': img_id,
                    'filename': img_info['file_name'],
                    'annotations': img_annotations[img_id]
                })
        
        return images_with_kpts
    
    def download_specific_images(self, images_list, split):
        """Télécharge des images spécifiques"""
        base_url = f"http://images.cocodataset.org/{split}/"
        target_dir = f"{self.base_dir}/images/{split}"
        os.makedirs(target_dir, exist_ok=True)
        
        print(f"📥 Téléchargement de {len(images_list)} images {split}...")
        
        for img_data in tqdm(images_list, desc=f"Images {split}"):
            filename = img_data['filename']
            img_url = base_url + filename
            img_path = f"{target_dir}/{filename}"
            
            if not os.path.exists(img_path):
                try:
                    urllib.request.urlretrieve(img_url, img_path)
                except Exception as e:
                    print(f"Erreur {filename}: {e}")
    
    def convert_to_yolo_format(self):
        """Convertit les annotations COCO au format YOLO"""
        print("🔄 Conversion au format YOLO...")
        
        from pycocotools.coco import COCO
        
        # Traiter train et val
        splits = [
            ('train2017', f"{self.base_dir}/annotations/person_keypoints_train2017.json"),
            ('val2017', f"{self.base_dir}/annotations/person_keypoints_val2017.json")
        ]
        
        for split, ann_file in splits:
            print(f"Conversion {split}...")
            coco = COCO(ann_file)
            
            # Images dans notre subset
            img_dir = f"{self.base_dir}/images/{split}"
            if not os.path.exists(img_dir):
                continue
                
            available_imgs = set(os.listdir(img_dir))
            
            label_dir = f"{self.base_dir}/processed/labels/{'train' if 'train' in split else 'val'}"
            img_out_dir = f"{self.base_dir}/processed/images/{'train' if 'train' in split else 'val'}"
            
            for img_filename in tqdm(available_imgs, desc=f"Conversion {split}"):
                if not img_filename.endswith('.jpg'):
                    continue
                    
                # Trouver l'image dans COCO
                img_id = None
                for coco_img_id, img_info in coco.imgs.items():
                    if img_info['file_name'] == img_filename:
                        img_id = coco_img_id
                        break
                
                if img_id is None:
                    continue
                
                # Copier l'image
                src_path = f"{img_dir}/{img_filename}"
                dst_path = f"{img_out_dir}/{img_filename}"
                shutil.copy2(src_path, dst_path)
                
                # Convertir annotations
                ann_ids = coco.getAnnIds(imgIds=img_id)
                anns = coco.loadAnns(ann_ids)
                
                label_filename = img_filename.replace('.jpg', '.txt')
                label_path = f"{label_dir}/{label_filename}"
                
                img_info = coco.imgs[img_id]
                img_w, img_h = img_info['width'], img_info['height']
                
                with open(label_path, 'w') as f:
                    for ann in anns:
                        if 'keypoints' in ann and len(ann['keypoints']) == 51:
                            # Convertir bbox
                            bbox = ann['bbox']
                            x_center = (bbox[0] + bbox[2]/2) / img_w
                            y_center = (bbox[1] + bbox[3]/2) / img_h
                            width = bbox[2] / img_w
                            height = bbox[3] / img_h
                            
                            # Convertir keypoints
                            keypoints = ann['keypoints']
                            kpts_normalized = []
                            
                            for i in range(0, len(keypoints), 3):
                                x = keypoints[i] / img_w if keypoints[i] > 0 else 0
                                y = keypoints[i+1] / img_h if keypoints[i+1] > 0 else 0
                                v = keypoints[i+2]
                                kpts_normalized.extend([x, y, v])
                            
                            kpts_str = ' '.join(map(str, kpts_normalized))
                            f.write(f"0 {x_center} {y_center} {width} {height} {kpts_str}\n")
        
        print("✅ Conversion YOLO terminée!")
    
    def create_dataset_yaml(self):
        """Crée le fichier de configuration YOLO"""
        config = {
            'path': os.path.abspath(f"{self.base_dir}/processed"),
            'train': 'images/train',
            'val': 'images/val',
            'kpt_shape': [17, 3],
            'names': {0: 'person'},
            'keypoints': [
                'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
            ],
            'skeleton': [
                [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
                [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
                [2, 4], [3, 5], [4, 6], [5, 7]
            ]
        }
        
        import yaml
        config_path = f"{self.base_dir}/dataset.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ Configuration sauvée: {config_path}")
        return config_path

def quick_setup_for_poc(max_train=500, max_val=100):
    """Setup rapide pour POC - subset de données"""
    print("🚀 Setup rapide COCO pour POC")
    
    # Installer dépendances
    try:
        from pycocotools.coco import COCO
    except ImportError:
        print("Installation de pycocotools...")
        os.system("pip install pycocotools")
    
    downloader = COCODownloader()
    
    # 1. Télécharger annotations (léger)
    downloader.download_annotations_only()
    
    # 2. Télécharger subset d'images (au lieu de 18GB)
    downloader.download_subset_images(max_train=max_train, max_val=max_val)
    
    # 3. Convertir au format YOLO
    downloader.convert_to_yolo_format()
    
    # 4. Créer fichier config
    config_path = downloader.create_dataset_yaml()
    
    print("🎉 Dataset COCO prêt pour l'entraînement!")
    print(f"📁 Dossier: {downloader.base_dir}")
    print(f"⚙️ Config: {config_path}")
    
    return downloader.base_dir, config_path

if __name__ == "__main__":
    # Setup rapide pour votre POC
    dataset_dir, config_file = quick_setup_for_poc()