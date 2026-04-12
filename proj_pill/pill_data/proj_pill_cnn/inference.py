import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import os
import json # 추가

try:
    from .model import PillNet  
except ImportError:
    from model import PillNet

class PillPredictor:
    def __init__(self, model_path="weights/pillnet_best.pth", label_map_path="label_map.json"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 라벨 맵 로드 및 클래스 수 자동 파악
        self.label_map = {}
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r', encoding='utf-8') as f:
                self.label_map = json.load(f)
            num_classes = len(self.label_map)
            print(f"📂 라벨 맵 로드 완료: {num_classes}개의 클래스")
        else:
            num_classes = 1000 # 기본값
            print("⚠️ 경고: label_map.json을 찾을 수 없습니다.")

        # 1. 모델 생성
        self.model = PillNet(num_classes=num_classes).to(self.device)
        
        # 2. 가중치 로드
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                print(f"✅ 모델 로드 성공: {model_path}")
            except Exception as e:
                print(f"❌ 모델 로드 실패: {e}")
        else:
            print(f"⚠️ 경고: 가중치 파일을 찾을 수 없습니다. ({model_path})")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image_bytes, threshold=90.0):
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_t = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(img_t)
                prob = F.softmax(outputs, dim=1)[0]
                top_probs, top_idxs = torch.topk(prob, 2)
                
                conf_score = float(top_probs[0].item()) * 100
                margin = (top_probs[0] - top_probs[1]).item() * 100
                class_idx = str(top_idxs[0].item()) # JSON 키는 문자열일 수 있음

            # 결과 반환 시 실제 클래스 이름 포함
            class_name = self.label_map.get(class_idx, "Unknown")

            if conf_score < threshold or margin < 20.0:
                return {
                    "success": True,
                    "need_retry": True,
                    "message": "확실하지 않습니다. 다시 촬영해 주세요.",
                    "confidence": conf_score,
                    "class_name": class_name # 어떤 건지 추측은 해주기
                }

            return {
                "success": True,
                "need_retry": False,
                "class_idx": int(class_idx),
                "class_name": class_name, # 실제 이름 반환
                "confidence": conf_score
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

if __name__ == "__main__":
    # 테스트 시에도 가급적 베스트 모델과 라벨 맵을 사용하세요.
    TEST_IMAGE = "sample_pill.png"
    TEST_WEIGHT = "weights_test/pillnet_best.pth" # 또는 학습된 에폭 파일
    
    predictor = PillPredictor(model_path=TEST_WEIGHT)
    
    if os.path.exists(TEST_IMAGE):
        with open(TEST_IMAGE, "rb") as f:
            img_data = f.read()
        print(f"🎯 예측 결과: {predictor.predict(img_data)}")