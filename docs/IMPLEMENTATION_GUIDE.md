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
- [x] No missing values
- [x] Timestamp 정렬됨
- [x] RPS, P99, Error rate 범위 검증 (이상치 제거)
- [x] Train/Warmup/Test 비율 확인 (70/10/20)

#### 2.2 시나리오 데이터 생성
**파일**: src/scenario_generator.py

```python
def create_normal_scenario(base_data: pd.DataFrame) -> pd.DataFrame:
    """기본 RPS 패턴 유지 (안정 구간)."""


def create_spike_scenario(..., spike_magnitude: float = 5.0, spike_duration: int = 60, ...):
    """특정 구간 RPS 급증 + error rate 상승."""


def create_gradual_scenario(..., base_rps: float = 500, target_rps: float = 3500, ...):
    """30s baseline → 90s 선형 증가 → 30s peak 유지."""


def create_periodic_scenario(..., period: int = 600, n_cycles: int = 3, ...):
    """저부하/고부하 반복, transition 플래그/period 메타데이터 포함."""
```

---

### Phase 3: 모델 구현 (4시간)

#### 3.1 LSTM 모델 구현
**파일**: src/lstm_model.py

```text
입력 피처: [rps, p99_latency, error_rate, cpu_percent]

학습 절차 요약:
1. `create_sequences(train_df, window=60, horizon=60)` → (X, y)
2. StandardScaler(feature-wise)로 X 정규화
3. PyTorch 모델: LSTM(64) → LSTM(32) → Dense(16) → Dense(60)
4. Optimizer: Adam(lr=0.001), Loss: MSE, Epoch: 50 + EarlyStopping(patience=5)
5. 수렴 검증: 마지막 loss / 최저 loss < 1.1, Val MAE < mean(RPS) × 0.2
6. 산출물 저장: `lstm_model.pt`, `scaler.pkl`, 학습 곡선
```

**개발 체크리스트**:
- [x] 시퀀스/레이블 생성 (window/horizon)
- [x] StandardScaler 적용
- [x] PyTorch 학습 및 EarlyStopping 구현
- [x] 수렴 검증
- [x] StratifiedSequenceDataset(시나리오 균형 샘플링) 옵션 검증
- [x] 모델/스케일러/곡선 저장

#### 3.2 LinUCB Agent 구현
**파일**: src/linucb_agent.py

```python
class LinUCBAgent:
    def __init__(
        self,
        action_space=None,
        alpha: float = 0.25,
        context_keys=None,
        alpha_decay: bool = False,
        decay_tau: float = 10_000.0,
        min_alpha: float = 0.05,
        regularization: float = 1.0,
    ):
        self.action_space = action_space or [
            300, 400, 500, 600, 800, 1000, 1200, 1500, 1800, 2100, 2500, 3000, 3600, 4200, 5000,
        ]
        self.context_keys = list(
            context_keys
            or ["rps", "error_rate", "cpu_percent", "rps_delta_5s", "rps_std_30s", "time_of_day_sin", "time_of_day_cos"]
        )
        self.d = len(self.context_keys)
        self.A = {idx: np.eye(self.d) * regularization for idx in range(len(self.action_space))}
        self.b = {idx: np.zeros(self.d) for idx in range(len(self.action_space))}
        self.alpha0 = alpha
        self.use_alpha_decay = alpha_decay
        self.decay_tau = decay_tau
        self.min_alpha = min_alpha

    def _extract_context(self, row: pd.Series) -> np.ndarray:
        # 스펙에서 정의한 컨텍스트 키를 활용해 정규화된 벡터 생성
        ...

    def select_action(self, context: np.ndarray) -> int:
        # θ_a^T x + α_t √(x^T A_a^{-1} x) 최대값 선택
        ...

    def update(self, context: np.ndarray, action_idx: int, reward: float) -> None:
        self.A[action_idx] += np.outer(context, context)
        self.b[action_idx] += reward * context
```

**개발 체크리스트**:
- [x] 컨텍스트 키 확장 및 정규화(`rps_delta_5s`, `rps_std_30s`, `time_of_day_*`)
- [x] 계층형 액션 공간 및 warmup regret curve 계산
- [x] α 감쇠 스케줄(√t 기반) 및 하한(min_alpha) 적용
- [x] warmup 이후 온라인 업데이트(시뮬레이션 루프에서 `update`) 유지

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

```text
run_simulation 구성요소:
- 입력: 모델 객체(LSTM/LinUCB/Static), 시나리오별 DataFrame(normal/spike/gradual/periodic)
- LSTM: 마지막 60초 [RPS,P99,Error,CPU] 윈도우 활용, horizon=60 예측 → throttle 산출
- LinUCB: `LinUCBAgent._extract_context`로 [rps, error_rate, cpu_percent, rps_delta_5s, rps_std_30s, time_of_day_*] 벡터 생성 → action 선택 → 즉시 업데이트
- 출력: SimulationResult(p99_latency, success_rate, stability_score, adaptation_time, predictive_mae, tracking_lag_seconds, detailed_results)
- Spike: 10초 이동평균으로 adaptation time 계산
- Gradual/Periodic: 예측 MAE, tracking lag, pattern recognition, proactive adjustment 산출
```

**개발 체크리스트**:
- [x] 모델 간 공통 인터페이스 확인
- [x] 시나리오별 메타데이터 (phase, is_transition 등) 유지
- [x] 추가 메트릭 (predictive_mae, tracking_lag, pattern_recognition) 기록

---

### Phase 5: 평가 및 통계 (1.5시간)

**파일**: src/evaluation.py

```python
def compute_metrics(detailed_results: pd.DataFrame, scenario: str) -> dict:
    """
    시뮬레이터 초별 로그를 기반으로 공통/시나리오별 메트릭을 계산한다.
    - 공통: success_rate, p99_latency, stability_score, adaptation_time(Spike)
    - Gradual: predictive_mae (LSTM), tracking_lag_seconds (LinUCB)
    - Periodic: pattern_recognition(autocorr), proactive_adjustment(전환 전후 error rate)
    - 모든 메트릭에 대해 95% CI 반환
    """

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
- [x] P99 계산 로직 검증
- [x] Success rate 계산 확인
- [x] Stability score 검증
- [x] 통계 함수 (ttest, Cohen's d) 확인

---

### Phase 6: 병렬 실행 스크립트 (1시간)

**파일**: experiments/run_all_scenarios.py

```text
실행 파라미터:
- 입력 CSV: burstgpt_timeseries + train/warmup/test 세트
- 기본 실행: 4 scenarios × 3 models × 10 seeds = 120 runs (옵션으로 축약 가능)
- CLI 옵션: --no-gradual / --no-periodic / --skip-lstm / --train-limit 등

파이프라인:
1. train/warmup/test 로드 (없으면 즉시 split)
2. 모델 준비: Static(threshold=P95), LinUCB(warmup 100k+ steps), LSTM 학습
3. Scenario 생성: normal/spike/gradual/periodic
4. 각 seed에 대해 run_simulation 실행 → JSON & 상세 CSV 저장
5. 실험 완료 후 statistical_analysis + visualization 자동 호출
```

**개발 체크리스트**:
- [ ] CLI 옵션 정리 및 문서화
- [ ] 재실행 시 이미 존재하는 결과 건너뛰기
- [ ] 통계/시각화 자동 실행

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
- [x] 환경 설정 완료
- [x] data_pipeline.py 작성 및 테스트
- [x] lstm_model.py 작성 및 테스트
- [x] linucb_agent.py 작성 및 테스트
- [x] baseline.py 작성
- [x] simulator.py 작성 및 테스트
- [x] evaluation.py 작성
- [x] 병렬 실행 스크립트 완성

### Day 3-4: 실험
- [ ] 최종 데이터 준비 완료
- [ ] 60회 병렬 실행 시작
- [ ] 모든 실행 완료 및 결과 저장
- [ ] 통계 분석 완료

### Day 5-7: 논문 작성
- [x] 시각화 완성
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
