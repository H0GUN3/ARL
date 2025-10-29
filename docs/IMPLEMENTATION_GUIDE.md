# 구현 가이드 (Implementation Guide)

## 단계별 구현 절차

### Phase 1: 환경 설정 (30분)

#### 1.1 의존성 설치
```bash
# 필수 패키지
pip install torch torchvision torchaudio
pip install pandas numpy scipy scikit-learn
pip install matplotlib seaborn
pip install tqdm

# 권장
pip install jupyterlab  # 개발 및 디버깅용
```

#### 1.2 폴더 구조 생성
```bash
# @docs/FOLDER_STRUCTURE.md 참고하여 생성
mkdir -p src data experiments results plots logs
```

#### 1.3 데이터 준비
```bash
# BurstGPT v1.1 다운로드
# https://github.com/HPMLL/BurstGPT/releases/tag/v1.1
# data/ 폴더에 CSV 저장
```

---

### Phase 2: 데이터 파이프라인 (2시간)

#### 2.1 데이터 로딩 및 검증
**파일**: src/data_pipeline.py

```python
def main():
    # 1단계: 원본 데이터 로드
    raw_data = load_burstgpt('data/burstgpt_v1.1.csv')
    print(f"Loaded {len(raw_data)} rows")

    # 2단계: 기본 검증
    assert 'timestamp' in raw_data.columns
    assert 'response_tokens' in raw_data.columns

    # 3단계: 1초 단위 집계
    ts_data = aggregate_to_timeseries(raw_data, interval=1)
    print(f"Aggregated to {len(ts_data)} seconds")

    # 4단계: 데이터 분할
    splits = split_data(ts_data)
    print(f"Train: {len(splits['train'])} | Warmup: {len(splits['warmup'])} | Test: {[len(t) for t in splits['test']]}")

    # 5단계: 저장
    save_timeseries(ts_data, 'data/burstgpt_timeseries.csv')
    print("✅ Data pipeline complete")
```

**체크리스트**:
- [ ] No missing values
- [ ] Timestamp 정렬됨
- [ ] RPS, P99, Error rate 범위 검증 (이상치 제거)
- [ ] Train/Warmup/Test 비율 확인 (70/10/20)

#### 2.2 시나리오 데이터 생성
**파일**: src/scenario_generator.py

```python
def create_spike_scenario(base_data: pd.DataFrame, spike_magnitude: float = 5.0) -> pd.DataFrame:
    """
    베이스 데이터에서 특정 구간에 Spike 주입

    Logic:
    1. 무작위 시작 시간 선택
    2. 10초 동안 RPS를 5배로 증가
    3. 30초 유지
    4. 원래대로 복구
    """
    pass

def create_normal_scenario(base_data: pd.DataFrame) -> pd.DataFrame:
    """
    원본 데이터 그대로 사용 (정규 시나리오)
    """
    return base_data.copy()
```

---

### Phase 3: 모델 구현 (4시간)

#### 3.1 LSTM 모델 구현
**파일**: src/lstm_model.py

```python
import torch
import torch.nn as nn

class LSTMNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, output_size=60):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # 마지막 타임스텝만
        return output

class LSTMPredictor:
    def __init__(self, window_size=60, hidden_units=128):
        self.window_size = window_size
        self.model = LSTMNet(hidden_size=hidden_units)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.scaler = StandardScaler()  # 정규화용

    def fit(self, train_data: pd.DataFrame, epochs=50, batch_size=32):
        # 구현 로직
        pass

    def predict(self, context: pd.DataFrame) -> np.ndarray:
        # 구현 로직
        pass
```

**개발 체크리스트**:
- [ ] 모델 초기화 확인
- [ ] 데이터 정규화 적용
- [ ] Loss curve 모니터링
- [ ] 수렴 확인 (마지막 10 epoch 차이 < 0.01)
- [ ] 모델 저장 (checkpoint)

#### 3.2 LinUCB Agent 구현
**파일**: src/linucb_agent.py

```python
class LinUCBAgent:
    def __init__(self, action_space=None, alpha=0.25, d=3):
        self.action_space = action_space or list(range(500, 5001, 100))
        self.alpha = alpha
        self.d = d  # context dimension

        # 각 arm마다 A, b 초기화
        self.A = {a: np.eye(d) for a in self.action_space}
        self.b = {a: np.zeros(d) for a in self.action_space}

    def warmup(self, warmup_data: pd.DataFrame) -> dict:
        # 10% 데이터에서 학습
        regret_curve = []

        for t in range(len(warmup_data)):
            context = self._extract_context(warmup_data.iloc[t])
            action = self.select_action(context)
            reward = self._compute_reward(warmup_data.iloc[t])

            self.update(context, action, reward)
            regret_curve.append(...)

        return {'regret_curve': regret_curve, 'convergence': ...}

    def select_action(self, context: np.ndarray) -> int:
        # LinUCB 식 구현
        pass

    def update(self, context: np.ndarray, action: int, reward: float):
        # A, b 업데이트
        self.A[action] += np.outer(context, context)
        self.b[action] += reward * context
```

**개발 체크리스트**:
- [ ] Context 추출 로직 확인
- [ ] Action 선택 로직 (UCB 식)
- [ ] Regret curve 감소 확인
- [ ] 파라미터 수렴 확인

#### 3.3 Static Baseline
**파일**: src/baseline.py

```python
class StaticRateLimiter:
    def __init__(self, train_data: pd.DataFrame):
        # P95 계산
        self.threshold = np.percentile(train_data['rps'], 95)

    def select_action(self, context: np.ndarray) -> int:
        # RPS를 throttle limit으로 사용
        return int(self.threshold)
```

---

### Phase 4: 평가 시뮬레이터 (2시간)

**파일**: src/simulator.py

```python
def run_simulation(model, test_data: pd.DataFrame, scenario: str, seed: int) -> dict:
    """
    모델을 테스트 데이터에서 평가
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    results = {
        'p99_latency': [],
        'success_rate': [],
        'rejected_count': 0,
        'total_count': 0
    }

    for t in range(len(test_data)):
        # Context 추출
        context = np.array([
            test_data.iloc[t]['rps'],
            test_data.iloc[t]['error_rate'],
            test_data.iloc[t]['cpu']  # 추정값
        ])

        # Action 선택
        if isinstance(model, LSTMPredictor):
            action = model.predict(...)  # RPS 예측
        elif isinstance(model, LinUCBAgent):
            action = model.select_action(context)  # Throttle limit
        else:  # Static
            action = model.select_action(context)

        # Reward 계산
        success = simulate_request(action, test_data.iloc[t]['rps'])
        results['success_rate'].append(success)

        if not success:
            results['rejected_count'] += 1
        results['total_count'] += 1

    # 메트릭 계산
    final_metrics = compute_metrics(results, scenario)
    return final_metrics
```

**개발 체크리스트**:
- [ ] 모든 모델의 인터페이스 동일
- [ ] Reward 계산 로직 확인
- [ ] 시나리오별 처리 (normal vs spike)

---

### Phase 5: 평가 및 통계 (1.5시간)

**파일**: src/evaluation.py

```python
def compute_metrics(simulation_results: dict, scenario: str) -> dict:
    """메트릭 계산"""
    success_rate = np.mean(simulation_results['success_rate'])

    # P99 계산 (시뮬레이션 결과에서)
    p99_latency = np.percentile(
        simulation_results['latencies'], 99
    )

    # Stability score (SLA 준수율)
    sla_compliant = sum(1 for l in simulation_results['latencies'] if l < 150)
    stability = sla_compliant / len(simulation_results['latencies'])

    if scenario == 'spike':
        # Adaptation time 계산
        spike_start = simulation_results['spike_start']
        recovery_time = find_recovery_time(
            simulation_results['success_rate'][spike_start:]
        )
        return {
            'p99_latency': p99_latency,
            'success_rate': success_rate,
            'stability_score': stability,
            'adaptation_time': recovery_time
        }
    else:
        return {
            'p99_latency': p99_latency,
            'success_rate': success_rate,
            'stability_score': stability
        }

def run_statistical_analysis(results_dir: str) -> dict:
    """모든 결과에 대해 통계 검증"""
    # 1. 결과 로드
    lstm_results = load_results(f'{results_dir}/lstm_*')
    linucb_results = load_results(f'{results_dir}/linucb_*')

    # 2. Paired t-test
    t_stat, p_value = scipy.stats.ttest_rel(lstm_results, linucb_results)

    # 3. Effect size (Cohen's d)
    effect_size = compute_cohens_d(lstm_results, linucb_results)

    return {
        'p_value': p_value,
        'significant': p_value < 0.05,
        't_statistic': t_stat,
        'effect_size': effect_size,
        'winner': 'LinUCB' if np.mean(linucb_results) > np.mean(lstm_results) else 'LSTM'
    }
```

**개발 체크리스트**:
- [ ] P99 계산 로직 검증
- [ ] Success rate 계산 확인
- [ ] Stability score 검증
- [ ] 통계 함수 (ttest, Cohen's d) 확인

---

### Phase 6: 병렬 실행 스크립트 (1시간)

**파일**: experiments/run_all_scenarios.py

```python
import multiprocessing as mp
from itertools import product

def run_single_experiment(params):
    model_name, scenario, seed = params

    # 모델 로드
    if model_name == 'lstm':
        model = LSTMPredictor()
        model.load('models/lstm_model.pkl')
    elif model_name == 'linucb':
        model = LinUCBAgent()
        model.load('models/linucb_agent.json')
    else:
        model = StaticRateLimiter(train_data)

    # 테스트 데이터 로드
    test_data = load_test_data(scenario)

    # 시뮬레이션 실행
    metrics = run_simulation(model, test_data, scenario, seed)

    # 결과 저장
    save_results(f'results/{model_name}_{scenario}_{seed}.json', metrics)
    return metrics

def main():
    # 병렬 실행 매개변수
    params = list(product(
        ['lstm', 'linucb', 'static'],  # 모델
        ['normal', 'spike'],            # 시나리오
        range(10)                       # seeds
    ))

    # 병렬 실행
    with mp.Pool(processes=8) as pool:
        results = pool.map(run_single_experiment, params)

    print(f"✅ {len(results)} experiments completed")

    # 통계 분석
    statistical_results = run_statistical_analysis('results')
    save_report(statistical_results)
```

**개발 체크리스트**:
- [ ] 병렬 처리 설정 (프로세스 수)
- [ ] 결과 저장 경로 확인
- [ ] 중단 후 재개 가능한 구조 (이미 실행된 항목 건너뛰기)

---

### Phase 7: 시각화 및 리포팅 (1시간)

**파일**: experiments/visualization.py

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_p99_comparison(results: dict):
    """모델별 P99 Latency 비교"""
    sns.boxplot(data=results, x='model', y='p99_latency')
    plt.savefig('plots/comparison_p99_boxplot.png', dpi=300, bbox_inches='tight')

def plot_convergence_curves():
    """수렴 곡선 (LSTM loss, LinUCB regret)"""
    pass

def generate_latex_tables(results: dict):
    """LaTeX 형식 결과 테이블"""
    pass
```

---

## 개발 체크리스트 (마스터 체크리스트)

### Day 1-2: 구현
- [ ] 환경 설정 완료
- [ ] data_pipeline.py 작성 및 테스트
- [ ] lstm_model.py 작성 및 테스트
- [ ] linucb_agent.py 작성 및 테스트
- [ ] baseline.py 작성
- [ ] simulator.py 작성 및 테스트
- [ ] evaluation.py 작성
- [ ] 병렬 실행 스크립트 완성

### Day 3-4: 실험
- [ ] 최종 데이터 준비 완료
- [ ] 60회 병렬 실행 시작
- [ ] 모든 실행 완료 및 결과 저장
- [ ] 통계 분석 완료

### Day 5-7: 논문 작성
- [ ] 시각화 완성
- [ ] 최종 리포트 생성
- [ ] 논문 초안 작성
- [ ] 검토 및 최종 제출

---

## 디버깅 팁

### 1. 메모리 부족
```python
# Batch size 감소
batch_size = 16  # 32에서 감소
```

### 2. NaN 손실값
```python
# 입력 정규화 확인
context = (context - context.mean()) / (context.std() + 1e-8)
```

### 3. 느린 실행
```python
# GPU 사용 확인
print(f"Using device: {device}")  # cuda이어야 함
```

### 4. 결과 재현 불가
```python
# 모든 시드 고정
np.random.seed(42)
torch.manual_seed(42)
```

