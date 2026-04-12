import os
#추가 코드 4월 10일
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # OpenMP 충돌 에러 방지용

from pill_classifier import *
from get_cli_args import get_cli_args
from pathlib import Path
from PIL import Image
# import osconda activate smart_pill



class Dataset_Dir(Dataset):
    def __init__(self, args, dir_dataset, transform=None, target_transform=None, run_phase='train'):
        self.args = args
        self.dir_dataset = dir_dataset
        self.transform = transform
        self.target_transform = target_transform

        self.list_images = [ png.name  for png in Path(dir_dataset).iterdir() if png.suffix == '.png']
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


job = 'resnet152'

# -> 기존 코드
# if __name__ == '__main__':
#     # job = 'hrnet_w64'
#     job = 'resnet152'
#     args = get_cli_args(job=job, run_phase='test', aug_level=0, dataclass='01')

#     print(f'model_path_in is {args.model_path_in}')

#     dir_testimage = r'.\dir_testimage'

#     args.dataset_valid = Dataset_Dir(args, dir_testimage, transform=transform_normalize, run_phase='test' if args.run_phase == 'test' else 'valid')
#     args.batch_size = len(args.dataset_valid)
#     args.verbose = False
#     print(f'valid dataset was loaded')

#     pill_classifier(args)

#     print(args.path_img)
#     print(args.list_preds)
#     print('job done')

# 변경 코드 -> 4월 10일
if __name__ == '__main__':
    job = 'resnet152'
    args = get_cli_args(job=job, run_phase='test', aug_level=0, dataclass='01')
    
    # 결과 보관함을 미리 만들어 에러를 방지합니다. 
    args.path_img = []
    args.list_preds = []

    dir_testimage = r'.\dir_testimage'
    # 분석할 이미지를 로드합니다. [cite: 199, 202]
    args.dataset_valid = Dataset_Dir(args, dir_testimage, transform=transform_normalize, run_phase='test')
    
    args.batch_size = len(args.dataset_valid)
    args.verbose = False
    print(f'valid dataset was loaded')
    
    # 모델을 실행하여 알약을 분류합니다. [cite: 158, 194]
    pill_classifier(args)

    # 최종 결과를 화면에 출력합니다. [cite: 203]
    print("분석한 파일:", args.path_img)
    print("예측한 번호:", args.list_preds)
    print('job done')