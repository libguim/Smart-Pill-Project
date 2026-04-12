### [가상환경 설정 (Conda)]

### 0. VS Code를 '관리자 권한'으로 실행하기

### 1. Conda 기존 환경 종료하기
conda deactivate

### 2. 프로젝트용 가상환경 만들기
conda create -n smart_pill python=3.10.12 -y

### 3. 환경 활성화하기
conda activate smart_pill

### 4. 필수 라이브러리 설치하기
### (주의: requirements.txt 파일이 있는 디렉토리에서 실행)
pip install -r requirements.txt

---

## 📂 프로젝트 폴더 구조 (Project Structure)

본 프로젝트는 대용량 AI 학습 데이터를 효율적으로 관리하기 위해 **MLOps 표준 구조**를 따릅니다.

```text
/SMART-PILL-PROJECT
├── 📂 data                # 모든 데이터가 집결되는 곳 (raw, processed 등 포함)
├── 📂 logs                # 학습 중 발생하는 로그, 손실(Loss) 기록 등이 저장되는 곳
├── 📂 models              # 학습된 결과물(가중치) 관리
│   └── 📄 .gitkeep        # 빈 폴더 유지를 위한 설정 파일
├── 📂 proj_pill\pill_data # [실제 학습 데이터셋]
│   ├── 📁 pill_data_croped       # 전처리가 완료된 전체 알약 이미지 (1,000종)
│   └── 📁 pill_data_croped_test  # 빠른 코드 검증을 위한 소량의 테스트 데이터
├── 📂 proj_pill_cnn       # [메인 실행 코드]
│   ├── 📄 model.py        # 모델 아키텍처 (PillNet 신경망 설계도)
│   ├── 📄 dataset.py      # 데이터 로딩 및 라벨 매핑 로직
│   ├── 📄 train.py        # 모델 학습 실행 엔진
│   ├── 📄 inference.py    # 실시간 웹캠 인식 서비스
│   └── 📁 weights         # 에폭별로 저장된 모델 가중치(.pth)
├── 📂 src                 # 프로젝트 공통 모듈 및 소스 코드 관리 폴더
├── .gitignore             # 가상환경, 대용량 데이터 등 Git 제외 목록 설정
├── README.md              # 프로젝트 매뉴얼 및 실행 방법 가이드
└── requirements.txt       # 프로젝트 실행에 필요한 라이브러리 목록