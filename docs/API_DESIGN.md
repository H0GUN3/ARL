# API 및 모듈 설계 (API & Module Design)

## 1. 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│         BurstGPT Raw Data                               │
│         (CSV, 5.29M traces)                             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ data_pipeline   │
            │ (Preprocessing) │
            └────────┬────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │ burstgpt_timeseries    │
        │ (1-sec aggregated)     │
        └────────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
    ▼             ▼             ▼
┌────────┐  ┌────────┐  ┌────────┐
│ LSTM   │  │LinUCB  │  │Static  │
│ Model  │  │ Agent  │  │ Agent  │
└────┬───┘  └───┬────┘  └───┬────┘
     │          │           │
     └──────────┼───────────┘
                │
                ▼
        ┌───────────────┐
        │  Simulator    │
        │  (Evaluation) │
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │  Evaluation   │
        │  Metrics      │
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │   Results     │
        │   (CSV)       │
        └───────────────┘
```

---

## 2. 모듈별 API 정의

### 2.1 data_pipeline.py

**목적**: BurstGPT 원본 데이터를 1초 단위 시계열로 변환

#### 함수: `load_burstgpt(csv_path: str) -> pd.DataFrame`
```python
"""
입력:
  csv_path: BurstGPT CSV 파일 경로

반환:
  DataFrame with columns:
    - timestamp (datetime)
    - request_tokens (int)
    - response_tokens (int)
    - model (str)
    - log_type (str)

예외:
  FileNotFoundError: 파일 없음
  ValueError: 데이터 형식 오류
"""
```

#### 함수: `aggregate_to_timeseries(df: pd.DataFrame, interval: int = 1) -> pd.DataFrame`
```python
"""
입력:
  df: 원본 BurstGPT DataFrame
  interval: 집계 간격 (초, 기본 1초)

반환:
  DataFrame with columns:
    - timestamp (datetime, index)
    - rps (float): requests per second
    - p99_latency (float): response_tokens × 30ms
    - error_rate (float): failed / total
    - avg_response_tokens (int)

계산 로직:
  rps = count(requests) / interval
  p99_latency = quantile(response_tokens, 0.99) × 30
  error_rate = count(response_tokens == 0) / count(all)
"""
```

#### 함수: `split_data(df: pd.DataFrame) -> dict`
```python
"""
입력:
  df: 1초 단위 시계열 DataFrame (121일)

반환:
  {
    'train': DataFrame (Day 1-84),
    'warmup': DataFrame (Day 85-96),
    'test': [DataFrame, ...] (Day 97-121, 5등분),
    'split_info': {
      'train_days': (1, 84),
      'warmup_days': (85, 96),
      'test_windows': [(97, 101), (102, 106), ...]
    }
  }

검증:
  ✅ No time leakage (test > warmup > train)
  ✅ 5개 test 구간 균등 분할
"""
```

#### 함수: `save_timeseries(df: pd.DataFrame, output_path: str) -> None`
```python
"""
입력:
  df: 1초 단위 시계열 DataFrame
  output_path: 저장 경로 (e.g., data/burstgpt_timeseries.csv)

저장 형식:
  CSV with columns: timestamp, rps, p99_latency, error_rate, avg_response_tokens
"""
```

---

### 2.2 lstm_model.py

**목적**: LSTM 기반 RPS 예측 모델

#### 클래스: `LSTMPredictor`

```python
class LSTMPredictor:
    """
    LSTM 모델로 미래 60초 RPS 예측

    속성:
      window_size: 입력 시계열 길이 (60초)
      hidden_units: LSTM 은닉층 크기 (128)
      dropout: Dropout rate (0.2)
      device: 실행 디바이스 ('cuda' or 'cpu')
    """
```

#### 메서드: `__init__(window_size: int = 60, hidden_units: int = 128, dropout: float = 0.2)`
```python
"""
초기화:
  - LSTM 신경망 구축
  - 모델을 device에 이동
"""
```

#### 메서드: `fit(train_data: pd.DataFrame, epochs: int = 50, batch_size: int = 32) -> dict`
```python
"""
입력:
  train_data: DataFrame with 'rps' column (1.76M 초, 70%)
  epochs: 학습 반복 수
  batch_size: 배치 크기

반환:
  {
    'train_loss': [float, ...],  # 각 epoch 손실값
    'convergence': bool,  # 수렴 확인 (마지막 10 epoch 손실 차 < 0.01)
    'model_path': str  # 저장된 모델 경로
  }

손실함수: MSE (Mean Squared Error)
옵티마이저: Adam (lr=0.001)
"""
```

#### 메서드: `predict(context: pd.DataFrame) -> np.ndarray`
```python
"""
입력:
  context: DataFrame with 'rps' column (최소 window_size = 60초)

반환:
  np.ndarray: 예측된 미래 60초 RPS (shape: (60,))

로직:
  1. 마지막 60초 데이터 추출
  2. 정규화 (훈련 시 사용한 평균/표준편차)
  3. 모델 통과
  4. 역정규화
"""
```

#### 메서드: `save(path: str) -> None` / `load(path: str) -> None`
```python
"""
경로에 모델 저장/로드
사용 형식: torch.save(), torch.load()
"""
```

---

### 2.3 linucb_agent.py

**목적**: LinUCB 기반 온라인 적응 Rate Limiting

#### 클래스: `LinUCBAgent`

```python
class LinUCBAgent:
    """
    Contextual Bandit 기반 Rate Limiter

    속성:
      action_space: 가능한 throttle 임계값들 (50개, 500-5000 RPS)
      exploration_rate: 탐험/활용 비율 (epsilon-greedy)
      alpha: 신뢰도 파라미터 (0.25, 기본값)
    """
```

#### 메서드: `__init__(action_space: list = None, alpha: float = 0.25)`
```python
"""
초기화:
  action_space: 기본값 [500, 600, ..., 5000] (50개 action)
  alpha: 신뢰도 파라미터

내부 상태:
  - A (d×d matrix): 컨텍스트 외적의 합
  - b (d vector): 리워드의 합
  - 초기값: 0으로 설정
"""
```

#### 메서드: `warmup(warmup_data: pd.DataFrame) -> dict`
```python
"""
입력:
  warmup_data: DataFrame (0.25M 초, 10%)

반환:
  {
    'regret_curve': [float, ...],  # 각 초의 누적 후회
    'convergence': bool,  # 수렴 확인
    'final_params': dict  # 학습된 파라미터
  }

동작:
  1. 매 초마다:
     - context 관찰 (rps, error_rate, cpu)
     - action 선택 (LinUCB 식)
     - reward 받음 (success_rate)
     - 파라미터 업데이트 (A, b)
"""
```

#### 메서드: `select_action(context: np.ndarray) -> int`
```python
"""
입력:
  context: [rps, error_rate, cpu] (shape: (3,))

반환:
  int: 선택된 action index (0-49)

로직:
  arm = argmax_a (θ_a^T x + α √(x^T A_a^{-1} x))
  여기서 θ_a = A_a^{-1} b_a
"""
```

#### 메서드: `update(context: np.ndarray, action: int, reward: float) -> None`
```python
"""
입력:
  context: [rps, error_rate, cpu]
  action: 선택된 action index
  reward: 즉각적인 보상 (success rate)

동작:
  1. A[action] ← A[action] + context @ context.T
  2. b[action] ← b[action] + reward * context
"""
```

#### 메서드: `save(path: str) -> None` / `load(path: str) -> None`
```python
"""
JSON 형식으로 A, b 행렬 저장/로드
"""
```

---

### 2.4 simulator.py

**목적**: Offline 환경에서 모든 모델 평가

#### 함수: `run_simulation(model, test_data: pd.DataFrame, scenario: str) -> dict`
```python
"""
입력:
  model: LSTMPredictor, LinUCBAgent, or StaticRateLimiter
  test_data: 평가용 DataFrame (0.5M 초, 20%)
  scenario: 'normal' or 'spike'

반환:
  {
    'p99_latency': float,  # 평균 P99 latency
    'success_rate': float,  # 거부되지 않은 요청 비율
    'stability_score': float,  # SLA 준수 시간 비율
    'adaptation_time': float or None,  # spike 시나리오만
    'detailed_results': pd.DataFrame  # 초별 결과
  }

동작 (pseudo-code):
  for t in test_data:
    if scenario == 'spike':
      rps = create_spike(t)
    else:
      rps = test_data[t].rps

    context = extract_context(rps, error_rate, cpu)
    action = model.predict(context)
    success = simulate_request(action, rps)

    결과 저장: p99, success_rate, stability
"""
```

#### 함수: `extract_context(rps: float, error_rate: float, cpu: float) -> np.ndarray`
```python
"""
입력:
  rps, error_rate, cpu: 현재 시스템 상태

반환:
  np.ndarray: 정규화된 context [0-1]
"""
```

#### 함수: `simulate_request(throttle_limit: int, current_rps: float) -> bool`
```python
"""
입력:
  throttle_limit: Rate limiter가 설정한 허용 RPS
  current_rps: 실제 들어오는 RPS

반환:
  bool: True (요청 통과) or False (거부)

로직:
  if current_rps <= throttle_limit:
    return True
  else:
    return False with probability (throttle_limit / current_rps)
"""
```

---

### 2.5 evaluation.py

**목적**: 메트릭 계산 및 통계 분석

#### 함수: `compute_metrics(detailed_results: pd.DataFrame, scenario: str) -> dict`
```python
"""
입력:
  detailed_results: 초별 결과 DataFrame
  scenario: 'normal' or 'spike'

반환:
  {
    'p99_latency': float,
    'success_rate': float,
    'stability_score': float,
    'adaptation_time': float or None,
    'confidence_interval': (float, float)  # 95% CI
  }
"""
```

#### 함수: `run_statistical_tests(model1_results: list, model2_results: list) -> dict`
```python
"""
입력:
  model1_results: 10회 반복 결과 [metric1, metric2, ...]
  model2_results: 10회 반복 결과

반환:
  {
    'p_value': float,  # paired t-test
    'significant': bool,  # p < 0.05
    'effect_size': float,  # Cohen's d
    't_statistic': float
  }

테스트:
  scipy.stats.ttest_rel(model1_results, model2_results)
"""
```

#### 함수: `generate_report(all_results: dict) -> str`
```python
"""
입력:
  all_results: 모든 모델×시나리오×seed 결과

반환:
  str: 마크다운 형식 최종 리포트

내용:
  - 시나리오별 성능 표
  - 통계 검증 결과
  - 시각화 파일 경로
"""
```

---

## 3. 데이터 흐름

### 3.1 학습 단계 (Training Phase)

```
Day 1-2:
  Train Data (70%) → data_pipeline → LSTM.fit() → model.pkl
  Warmup Data (10%) → data_pipeline → LinUCB.warmup() → agent.json
```

### 3.2 평가 단계 (Evaluation Phase)

```
Day 3-4:
  Test Data (20%, 5 windows) → Simulator
  ├─ LSTM model → metrics
  ├─ LinUCB agent → metrics
  └─ Static baseline → metrics

  10 seeds × 2 scenarios = 60 runs in parallel
  ↓
  results/ → statistical_analysis
```

### 3.3 분석 단계 (Analysis Phase)

```
Day 5-7:
  results/ → evaluation.py → statistical_tests
  ├─ t-test
  ├─ effect size
  └─ confidence intervals
  ↓
  plots/ (visualization)
  ↓
  논문 작성
```

---

## 4. 에러 처리

### 4.1 Data Pipeline
| 에러 | 처리 |
|------|------|
| FileNotFoundError | 상세 메시지와 함께 종료 |
| Missing columns | 필수 컬럼 목록 출력 |
| Time leakage | 검증 후 assert 실패 |

### 4.2 Model Training
| 에러 | 처리 |
|------|------|
| OOM (Out of Memory) | batch_size 감소 후 재시작 |
| NaN loss | 학습률 감소 또는 데이터 정규화 |
| Convergence fail | warning 출력, 모델 저장 |

### 4.3 Evaluation
| 에러 | 처리 |
|------|------|
| Model not found | 명확한 에러 메시지 |
| Shape mismatch | 입력 검증 강화 |

---

## 5. 함수 서명 요약

| 모듈 | 함수 | 입력 | 출력 |
|------|------|------|------|
| data_pipeline | load_burstgpt | str | DataFrame |
| data_pipeline | aggregate_to_timeseries | DataFrame | DataFrame |
| data_pipeline | split_data | DataFrame | dict |
| lstm_model | fit | DataFrame | dict |
| lstm_model | predict | DataFrame | ndarray |
| linucb_agent | warmup | DataFrame | dict |
| linucb_agent | select_action | ndarray | int |
| simulator | run_simulation | model, DataFrame | dict |
| evaluation | compute_metrics | DataFrame | dict |
| evaluation | run_statistical_tests | list, list | dict |

---

## 6. 확장성 고려사항

- **병렬 실행**: 60개 run을 병렬로 실행하기 위해 각 함수는 state를 공유하지 않도록 설계
- **결과 저장**: 중간 결과를 자동으로 저장하여 재실행 불필요
- **재현성**: 모든 seed와 하이퍼파라미터 로깅

