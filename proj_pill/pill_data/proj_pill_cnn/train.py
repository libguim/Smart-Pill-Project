import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import numpy as np
from model import PillNet
from dataset import get_dataloader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns

def train_model():
    # 1. 환경 및 메모리 설정
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # GPU 메모리 찌꺼기 제거
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 실행 경로에 'test'가 포함되어 있는지 확인하여 모드 결정
    is_test_mode = "test" in os.getcwd().lower()
    print(f"🚀 {'🧪 테스트' if is_test_mode else '🔥 본학습'} 모드 시작 ({device})")

    # 2. 하이퍼파라미터 및 경로 설정
    if is_test_mode:
        BATCH_SIZE = 32 if torch.cuda.is_available() else 16
        num_epochs = 5  # 테스트는 빠르게 확인만
        root_dir = "../pill_data_croped_test"
        weight_dir = "weights_test"
    else:
        BATCH_SIZE = 64 if torch.cuda.is_available() else 8
        num_epochs = 30
        root_dir = "../pill_data_croped"
        weight_dir = "weights"

    # 데이터 로더 로드
    train_loader, num_classes = get_dataloader(root_dir, batch_size=BATCH_SIZE)
    os.makedirs(weight_dir, exist_ok=True)

    # [보완] 추론 시 혼란을 방지하기 위해 클래스 개수 기록
    with open(os.path.join(weight_dir, "class_info.txt"), "w") as f:
        f.write(f"num_classes: {num_classes}")

    # 3. 모델, 손실함수, 최적화 도구 설정
    model = PillNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5에폭마다 학습률을 1/10로 감소시켜 정밀도 향상
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # 4. 학습 루프 변수 설정
    history = {'loss': [], 'top1_acc': []}
    best_loss = float('inf') # 최저 손실값 기록용

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []
        top5_correct, total_samples = 0, 0
        
        current_lr = optimizer.param_groups[0]['lr']
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [LR: {current_lr:.6f}]")
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 성능 지표 데이터 수집
            with torch.no_grad():
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                if num_classes >= 5:
                    _, top5_preds = outputs.topk(5, 1, True, True)
                    top5_correct += (top5_preds == labels.view(-1, 1)).sum().item()
                    total_samples += labels.size(0)

            pbar.set_postfix({'loss': running_loss / len(train_loader)})

        # 에폭 종료 후 스케줄러 업데이트
        scheduler.step()

        # 에폭 통계 계산
        epoch_loss = running_loss / len(train_loader)
        top1_acc = (np.array(all_preds) == np.array(all_labels)).mean() * 100
        history['loss'].append(epoch_loss)
        history['top1_acc'].append(top1_acc)

        print(f"📊 Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Acc: {top1_acc:.2f}%")

        # [핵심 보완] 모든 에폭 저장 + Best 모델 별도 저장
        save_path = os.path.join(weight_dir, f"pillnet_v1_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = os.path.join(weight_dir, "pillnet_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"⭐ Best Model 갱신 (Loss: {best_loss:.4f}) -> {best_path}")

    # 5. 혼동 행렬 시각화
    if not is_test_mode and num_classes <= 50:
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(all_labels, all_preds)
        sns.heatmap(cm, annot=False, cmap='Blues')
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(weight_dir, "confusion_matrix.png"))

    # 6. 학습 곡선 저장
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), history['loss'], label='Loss', color='blue', marker='o')
    plt.title(f"Training Loss Curve (Final LR: {optimizer.param_groups[0]['lr']:.6f})")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    
    graph_name = 'test_learning.png' if is_test_mode else 'learning_curve.png'
    plt.savefig(graph_name) 
    print(f"📈 결과 리포트 저장 완료: {graph_name}")
    
    if not is_test_mode:
        plt.show()

if __name__ == "__main__":
    train_model()