import os
import torch
import torch.nn as nn
from torchvision import models

class PillNet(nn.Module):
    def __init__(self, num_classes=1000, use_pretrained=True):
        super(PillNet, self).__init__()
        
        # [체크] weights='DEFAULT'는 최신 버전인 IMAGENET1K_V2 등을 자동으로 가져옵니다.
        if use_pretrained:
            self.backbone = models.resnet50(weights='DEFAULT')
            print("🚀 [전이 학습] ResNet50 사전 학습 가중치 로드 완료")
        else:
            self.backbone = models.resnet50(weights=None)
            print("🌱 [신규 학습] 모델 구조 초기화 완료")

        # ResNet50의 마지막 fc 레이어 입력 피처 수 (2048)
        in_features = self.backbone.fc.in_features

        # [핀셋 보완] 분류기 구조
        # 1024 정도로 한 번 더 걸러주면 복잡한 알약 문양을 더 잘 파악할 수 있습니다.
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 1024), 
            nn.BatchNorm1d(1024), # 학습 안정성을 위한 배치 정규화 추가
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        # x: (Batch, 3, 224, 224)
        return self.backbone(x)

if __name__ == "__main__":
    # 실행 시 num_classes가 0이 되지 않도록 방어 로직 추가
    test_data_path = "../pill_data_croped_test"
    num_cls = 1000
    
    if os.path.exists(test_data_path):
        dirs = [d for d in os.listdir(test_data_path) if os.path.isdir(os.path.join(test_data_path, d))]
        if len(dirs) > 0:
            num_cls = len(dirs)
        
    model = PillNet(num_classes=num_cls)
    
    # 추론 모드로 변경하여 BatchNorm 에러 방지
    # eval() 모드에서는 BatchNorm이 학습된 통계치를 사용하므로 데이터가 1개여도 에러가 나지 않습니다.
    model.eval() 
    
    # 실제 데이터가 들어왔을 때를 가정한 테스트
    with torch.no_grad(): # 테스트 시 메모리 효율을 위해 추가 권장
        sample_input = torch.randn(1, 3, 224, 224)
        output = model(sample_input)
    
    print(f"📊 최종 클래스 수: {num_cls}")
    print(f"🎯 출력 텐서 형태: {output.shape}")