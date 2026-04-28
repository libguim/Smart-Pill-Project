import os
import json
# 추가 코드 4월 10일
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # OpenMP 충돌 에러 방지용

from pill_classifier import *
from get_cli_args import get_cli_args
from pathlib import Path
from PIL import Image

class Dataset_Dir(Dataset):
    def __init__(self, args, dir_dataset, transform=None, target_transform=None, run_phase='train'):
        self.args = args
        self.dir_dataset = dir_dataset
        self.transform = transform
        self.target_transform = target_transform

        # .png와 .jpg 파일을 모두 인식하도록 수정 
        self.list_images = [ f.name for f in Path(dir_dataset).iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        self.run_phase = run_phase

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.dir_dataset, self.list_images[idx]))
        label = 0
        path_img = self.list_images[idx]
        aug_name = ""

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        if self.run_phase == 'valid' or self.run_phase == 'test':
            return image, label, path_img, aug_name
        else:
            return image, label

if __name__ == '__main__':
    job = 'resnet152'
    args = get_cli_args(job=job, run_phase='test', aug_level=0, dataclass='01')
    
    args.path_img = []
    args.list_preds = []

    dir_testimage = r'.\dir_testimage'
    args.dataset_valid = Dataset_Dir(args, dir_testimage, transform=transform_normalize, run_phase='test')
    
    if len(args.dataset_valid) == 0:
        print("분석할 이미지가 폴더에 없습니다.")
    else:
        args.batch_size = len(args.dataset_valid)
        args.verbose = False
        print(f'데이터 로드 완료. 분석 시작...')
        
        # 모델 실행 [cite: 194]
        pill_classifier(args)

        # --- ID 번호(K-xxxxxx) 매칭 로직 추가 ---
        # JSON 파일을 읽어 라벨 번호와 알약 ID를 연결합니다. 
        with open(args.json_pill_label_path_sharp_score, 'r', encoding='utf-8') as f:
            label_data = json.load(f)
            # label_data["pill_label_path_sharp_score"]는 [[0, "K-037589", ...], [1, "K-029534", ...]] 형태입니다. [cite: 28]
            id_mapping = {item[0]: item[1] for item in label_data["pill_label_path_sharp_score"]}

        # 예측된 번호를 K-번호로 변환
        predicted_ids = [id_mapping.get(pred, "알 수 없음") for pred in args.list_preds]

        # 최종 결과 출력 
        print("\n" + "="*50)
        print("분석한 파일명:", args.path_img)
        print("예측된 알약 ID:", predicted_ids)
        print("="*50)
        print('job done')