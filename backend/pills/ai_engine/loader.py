import torch
from .pill_classifier import get_pill_model
from .utils import model_load, transform_normalize
from .get_cli_args import get_cli_args

def get_deployed_model():
    # 1. 학습 때 썼던 인자 그대로 생성 (run_phase='test')
    args = get_cli_args(job='resnet152', run_phase='test')
    
    # 2. 모델 구조 선언
    model = get_pill_model(args)
    
    # 3. 가중치 로드 (utils.py의 로직 활용)
    # 기존 코드 수정 없이 args.model_path_in 위치만 맞춰주면 됩니다.
    checkpoint = torch.load(args.model_path_in, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        # 현재 에러 메시지상 가중치는 'model' 키 안에 들어있습니다.
        model.load_state_dict(checkpoint['model'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # 파일 자체가 state_dict인 경우
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    return model, transform_normalize