import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# 1. 장고 환경 감지
try:
    from django.conf import settings
    IS_DJANGO = True
except ImportError:
    IS_DJANGO = False

class PillDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # [클래스 수집 로직]
        self.classes = sorted([d for d in os.listdir(root_dir) 
                               if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for i, cls_name in enumerate(self.classes)}

        # [이중 저장 경로 설정]
        save_paths = []
        
        # 1) 현재 실행 위치에 저장 (로컬 확인용)
        save_paths.append('label_map.json')
        
        # 2) 장고 환경일 경우 추가 경로 확보
        if IS_DJANGO:
            # settings.py에 LABEL_MAP_PATH가 설정되어 있다면 해당 경로 추가
            django_path = getattr(settings, 'LABEL_MAP_PATH', None)
            if django_path:
                save_paths.append(django_path)
            else:
                # 설정이 없다면 기본적으로 BASE_DIR/models/label_map.json 시도
                default_django_path = os.path.join(settings.BASE_DIR, 'models', 'label_map.json')
                save_paths.append(default_django_path)

        # [파일 저장 실행]
        for path in save_paths:
            try:
                # 폴더가 없으면 생성 (경로가 'label_map.json'이면 현재 폴더이므로 에러 안 남)
                dir_name = os.path.dirname(os.path.abspath(path))
                os.makedirs(dir_name, exist_ok=True)
                
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(self.idx_to_class, f, ensure_ascii=False, indent=4)
                print(f"✅ 라벨 맵 저장 완료: {path}")
            except Exception as e:
                print(f"⚠️ {path} 저장 중 오류 발생: {e}")

        # [이하 데이터 로드 로직 동일]
        self.image_paths = []
        self.labels = []
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(valid_extensions):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_num_classes(self):
        return len(self.classes)

# [데이터로더 함수 동일]
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_dataloader(root_dir, batch_size=32):
    dataset = PillDataset(root_dir, transform=data_transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, dataset.get_num_classes()