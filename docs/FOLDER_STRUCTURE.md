# 폴더 구조 정의 (Folder Structure)

## 1. 최종 폴더 구조

```
limiting/
│
├── CLAUDE.md                          # 스펙 주도 개발 가이드 (메인)
├── 검증.md                            # 방법론 검증 문서
├── AI문제.md                          # AI 피드백
├── README.md                          # 프로젝트 개요
│
├── docs/                              # 📋 명세 및 설계 문서
│   ├── SPEC.md                        # 프로젝트 명세
│   ├── API_DESIGN.md                  # API 및 모듈 설계
│   ├── IMPLEMENTATION_GUIDE.md         # 구현 가이드
│   ├── TESTING_STRATEGY.md            # 테스트 전략
│   ├── AI_INTEGRATION.md              # AI 도구 사용법
│   ├── FOLDER_STRUCTURE.md            # 이 파일
│   └── REFERENCES.md                  # 참고 자료
│
├── src/                               # 🔨 소스 코드
│   ├── __init__.py
│   ├── data_pipeline.py               # 데이터 전처리
│   ├── lstm_model.py                  # LSTM 모델
│   ├── linucb_agent.py                # LinUCB 에이전트
│   ├── baseline.py                    # Static baseline
│   ├── simulator.py                   # 오프라인 시뮬레이터
│   ├── evaluation.py                  # 메트릭 계산
│   ├── scenario_generator.py          # 시나리오 생성
│   └── utils.py                       # 유틸리티 함수
│
├── tests/                             # ✅ 테스트 코드
│   ├── __init__.py
│   ├── test_data_pipeline.py
│   ├── test_lstm_model.py
│   ├── test_linucb_agent.py
│   ├── test_simulator.py
│   ├── test_evaluation.py
│   ├── test_integration.py            # End-to-end 테스트
│   ├── test_performance.py            # 성능 테스트
│   ├── test_fairness.py               # 공정성 검증
│   ├── test_statistics.py             # 통계 검증
│   └── test_regression.py             # 회귀 테스트
│
├── experiments/                       # 🧪 실험 및 분석
│   ├── __init__.py
│   ├── run_all_scenarios.py           # 60회 병렬 실행
│   ├── statistical_analysis.py        # 통계 분석
│   ├── visualization.py               # 시각화
│   └── notebooks/                     # Jupyter 노트북 (선택)
│       └── exploration.ipynb
│
├── data/                              # 📊 데이터
│   ├── README.md                      # 데이터 설명
│   ├── burstgpt_v1.1.csv              # 원본 (다운로드 필요)
│   └── burstgpt_timeseries.csv        # 전처리된 (생성)
│
├── models/                            # 💾 학습된 모델
│   ├── lstm_model.pkl                 # LSTM 체크포인트
│   ├── lstm_model_final.pkl           # 최종 LSTM
│   ├── linucb_agent.json              # LinUCB 파라미터
│   └── metadata.json                  # 모델 메타데이터
│
├── results/                           # 📈 실험 결과
│   ├── scenario_1_normal_results.csv  # Normal 시나리오 결과
│   ├── scenario_2_spike_results.csv   # Spike 시나리오 결과
│   ├── statistical_summary.txt        # 통계 요약
│   ├── convergence_analysis.json      # 수렴 분석
│   └── comparison_table.csv           # 모델 비교 표
│
├── plots/                             # 📊 시각화
│   ├── comparison_p99_boxplot.png
│   ├── success_rate_barplot.png
│   ├── stability_trend.png
│   ├── adaptation_time_histogram.png
│   ├── convergence_curves.png
│   ├── lstm_loss_curve.png
│   ├── linucb_regret_curve.png
│   └── results_summary.pdf
│
├── logs/                              # 📝 로그
│   ├── training_lstm.log
│   ├── warmup_linucb.log
│   ├── simulation_normal.log
│   ├── simulation_spike.log
│   └── analysis.log
│
├── .github/                           # CI/CD
│   └── workflows/
│       └── test.yml                   # GitHub Actions 설정
│
├── requirements.txt                   # Python 의존성
├── setup.py                           # 패키지 설정
├── .gitignore                         # Git 무시 파일
└── LICENSE                            # 라이선스
```

---

## 2. 주요 디렉토리 설명

### 2.1 docs/ - 명세 및 설계 문서

**목적**: 스펙 주도 개발을 위한 모든 명세 문서

| 파일 | 용도 | 작성자 | 주기 |
|------|------|--------|------|
| SPEC.md | 프로젝트 전체 명세 | 사용자 | 1회 |
| API_DESIGN.md | 모듈/함수 인터페이스 | 사용자 | 1회 |
| IMPLEMENTATION_GUIDE.md | 구현 절차 | 사용자 | 1회 |
| TESTING_STRATEGY.md | 테스트 계획 | 사용자 | 1회 |
| AI_INTEGRATION.md | Claude Code 사용법 | 사용자 | 1회 |

**사용법**:
```
# Claude Code에서 참조
"@docs/SPEC.md를 보고 구현해줘"
```

### 2.2 src/ - 구현 코드

**목적**: 프로젝트의 모든 파이썬 모듈

| 파일 | 목적 | 라인 | 의존성 |
|------|------|------|--------|
| data_pipeline.py | 데이터 전처리 | ~200 | pandas, numpy |
| lstm_model.py | LSTM 모델 | ~300 | torch, sklearn |
| linucb_agent.py | LinUCB 에이전트 | ~250 | numpy, scipy |
| baseline.py | Static baseline | ~50 | numpy |
| simulator.py | 오프라인 시뮬레이터 | ~200 | pandas, numpy |
| evaluation.py | 메트릭 계산 | ~200 | numpy, scipy |
| utils.py | 유틸리티 | ~100 | 기본 라이브러리 |

**구현 순서**:
1. data_pipeline.py
2. lstm_model.py
3. linucb_agent.py
4. baseline.py
5. simulator.py
6. evaluation.py

### 2.3 tests/ - 테스트 코드

**목적**: 모든 모듈의 단위/통합/성능/회귀 테스트

| 파일 | 테스트 타입 | 커버리지 |
|------|-----------|---------|
| test_data_pipeline.py | Unit | data_pipeline.py |
| test_lstm_model.py | Unit | lstm_model.py |
| test_linucb_agent.py | Unit | linucb_agent.py |
| test_integration.py | Integration | src/ 전체 |
| test_performance.py | Performance | 학습/추론 속도 |
| test_fairness.py | Validation | 공정성 기준 |
| test_statistics.py | Statistical | 통계 함수 |
| test_regression.py | Regression | 재현성 |

**실행**:
```bash
# 전체 테스트
pytest tests/ -v

# 특정 테스트
pytest tests/test_lstm_model.py -v

# Coverage 리포트
pytest tests/ --cov=src
```

### 2.4 experiments/ - 실험 스크립트

**목적**: 실험 실행 및 분석

| 파일 | 용도 | 입력 | 출력 |
|------|------|------|------|
| run_all_scenarios.py | 60회 병렬 실행 | src/모델들 | results/*.csv |
| statistical_analysis.py | 통계 검증 | results/*.csv | statistical_summary.txt |
| visualization.py | 시각화 생성 | results/*.csv | plots/*.png |

**실행 순서**:
```bash
# 1단계: 데이터 준비
python src/data_pipeline.py

# 2단계: 모델 학습
python experiments/train_models.py

# 3단계: 실험 실행
python experiments/run_all_scenarios.py

# 4단계: 분석
python experiments/statistical_analysis.py

# 5단계: 시각화
python experiments/visualization.py
```

### 2.5 data/ - 데이터

**원본 데이터**:
- burstgpt_v1.1.csv (5.29M rows, ~2GB)
- GitHub에서 다운로드 필요

**전처리 데이터**:
- burstgpt_timeseries.csv (10.4M rows, 121일 × 86400초)
- data_pipeline.py에서 자동 생성

### 2.6 models/ - 학습된 모델

**LSTM 모델**:
- lstm_model.pkl (최종 모델)
- lstm_model_checkpoint_*.pkl (체크포인트)

**LinUCB 에이전트**:
- linucb_agent.json (A, b 파라미터)

**메타데이터**:
```json
{
  "model_name": "lstm",
  "training_date": "2025-10-29",
  "train_loss": 0.0089,
  "convergence": true,
  "hyperparameters": {
    "window_size": 60,
    "hidden_units": 128,
    "epochs": 50
  }
}
```

### 2.7 results/ - 실험 결과

**CSV 결과 파일**:
```
model,scenario,seed,time_window,p99_latency,success_rate,stability_score,adaptation_time
lstm,normal,0,1,450.5,0.98,0.92,
lstm,normal,0,2,455.2,0.97,0.91,
...
linucb,spike,0,1,380.2,0.95,0.88,25.3
linucb,spike,0,1,385.1,0.94,0.87,26.1
...
```

**통계 요약**:
```
Model Comparison Results
========================

Scenario: Normal
  LSTM P99:     450.2 ± 25.3 ms
  LinUCB P99:   385.5 ± 28.1 ms
  p-value:      0.042 (significant)
  Effect size:  0.68 (medium)

Scenario: Spike
  LSTM Success Rate:     0.94 ± 0.05
  LinUCB Success Rate:   0.96 ± 0.03
  p-value:      0.156 (not significant)
  Adaptation Time: LinUCB 22±3s vs LSTM 35±5s
```

### 2.8 plots/ - 시각화

**생성되는 그래프**:
1. Comparison_p99_boxplot.png - P99 Latency 비교
2. success_rate_barplot.png - Success Rate 비교
3. stability_trend.png - Stability 시간 추이
4. adaptation_time_histogram.png - Spike 적응 시간
5. convergence_curves.png - LSTM loss & LinUCB regret
6. lstm_loss_curve.png - LSTM 학습 곡선
7. linucb_regret_curve.png - LinUCB 후회 곡선

### 2.9 logs/ - 로그 파일

**로그 레벨**:
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/{module}.log'),
        logging.StreamHandler()
    ]
)
```

**로그 내용**:
- 데이터 로딩: 행 수, 메모리 사용량
- 모델 학습: epoch, loss, 시간
- 실험 실행: seed, 진행률, 예상 시간
- 분석 완료: 통계 결과

---

## 3. 파일 생성 스크립트

### 3.1 폴더 구조 자동 생성

**스크립트**: create_folders.sh

```bash
#!/bin/bash

# 폴더 생성
mkdir -p limiting/{docs,src,tests,experiments/notebooks,data,models,results,plots,logs}

# docs 파일은 이미 생성됨

# src 초기 파일
touch limiting/src/{__init__.py,data_pipeline.py,lstm_model.py,linucb_agent.py,baseline.py,simulator.py,evaluation.py,scenario_generator.py,utils.py}

# tests 초기 파일
touch limiting/tests/{__init__.py,test_data_pipeline.py,test_lstm_model.py,test_linucb_agent.py,test_simulator.py,test_evaluation.py,test_integration.py,test_performance.py,test_fairness.py,test_statistics.py,test_regression.py}

# experiments 초기 파일
touch limiting/experiments/{__init__.py,run_all_scenarios.py,statistical_analysis.py,visualization.py}

# 최상위 파일
touch limiting/{README.md,requirements.txt,setup.py,.gitignore}

# .github 설정
mkdir -p limiting/.github/workflows
touch limiting/.github/workflows/test.yml

echo "✅ 폴더 구조 생성 완료!"
```

**실행**:
```bash
bash create_folders.sh
```

### 3.2 requirements.txt

```
# Core
python>=3.9

# Data Processing
pandas==2.0.3
numpy==1.24.3

# Deep Learning
torch==2.0.1
torchvision==0.15.2

# Scientific Computing
scipy==1.11.1
scikit-learn==1.3.0

# Visualization
matplotlib==3.7.2
seaborn==0.12.2

# Utilities
tqdm==4.66.1
pyyaml==6.0

# Testing
pytest==7.4.0
pytest-cov==4.1.0

# Development
jupyter==1.0.0
ipython==8.14.0
```

**설치**:
```bash
pip install -r requirements.txt
```

---

## 4. 파일 생성 체크리스트

### 4.1 문서 파일 (docs/)
- [x] SPEC.md
- [x] API_DESIGN.md
- [x] IMPLEMENTATION_GUIDE.md
- [x] TESTING_STRATEGY.md
- [x] AI_INTEGRATION.md
- [x] FOLDER_STRUCTURE.md (현재 파일)
- [ ] REFERENCES.md (향후)

### 4.2 소스 코드 (src/)
- [ ] __init__.py
- [ ] data_pipeline.py
- [ ] lstm_model.py
- [ ] linucb_agent.py
- [ ] baseline.py
- [ ] simulator.py
- [ ] evaluation.py
- [ ] scenario_generator.py
- [ ] utils.py

### 4.3 테스트 코드 (tests/)
- [ ] __init__.py
- [ ] test_data_pipeline.py
- [ ] test_lstm_model.py
- [ ] test_linucb_agent.py
- [ ] test_simulator.py
- [ ] test_evaluation.py
- [ ] test_integration.py
- [ ] test_performance.py
- [ ] test_fairness.py
- [ ] test_statistics.py
- [ ] test_regression.py

### 4.4 실험 스크립트 (experiments/)
- [ ] __init__.py
- [ ] run_all_scenarios.py
- [ ] statistical_analysis.py
- [ ] visualization.py

### 4.5 설정 파일
- [ ] requirements.txt
- [ ] setup.py
- [ ] .gitignore
- [ ] README.md

### 4.6 CI/CD
- [ ] .github/workflows/test.yml

---

## 5. 파일 크기 예상

| 디렉토리 | 파일 개수 | 예상 크기 |
|---------|----------|---------|
| docs/ | 6 | 150 KB |
| src/ | 9 | 200 KB |
| tests/ | 11 | 300 KB |
| experiments/ | 4 | 100 KB |
| data/ | 2 | 2 GB (원본) + 500 MB (전처리) |
| models/ | 3 | 50 MB |
| results/ | 5 | 100 MB |
| plots/ | 8 | 50 MB |
| **총계** | **48** | **~2.6 GB** |

---

## 6. Git 관리

### 6.1 .gitignore

```
# 데이터 (크기가 크므로)
data/burstgpt_v1.1.csv
data/burstgpt_timeseries.csv

# 모델 (재학습 가능)
models/*.pkl
models/*.pth

# 결과 (재실행 가능)
results/
plots/
logs/

# Python
__pycache__/
*.pyc
.pytest_cache/
.coverage
htmlcov/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
```

### 6.2 커밋 전략

```
Commit 1: docs/ - 모든 명세 작성
Commit 2: src/ - 기본 구현 (data_pipeline 포함)
Commit 3: src/ - LSTM 모델 구현
Commit 4: src/ - LinUCB 에이전트 구현
Commit 5: src/ - Simulator & Evaluation
Commit 6: tests/ - 모든 테스트 코드
Commit 7: experiments/ - 실험 스크립트
Commit 8: results/ & plots/ - 실험 결과
```

---

## 다음 단계

> 이제 폴더 구조를 생성하고 순서대로 구현을 시작하면 됩니다!

**준비 체크리스트**:
1. [ ] docs/ 모든 파일 완성 (이미 완료)
2. [ ] 폴더 구조 생성 (아래 명령어)
3. [ ] requirements.txt 설치
4. [ ] BurstGPT 데이터 다운로드
5. [ ] src/data_pipeline.py부터 순차 구현

