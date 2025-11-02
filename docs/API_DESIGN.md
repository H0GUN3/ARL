# API 및 모듈 설계 (API & Module Design)

## 1. 시스템 아키텍처

```
BurstGPT Raw Data (CSV)
        │
        ▼
data_pipeline.py ──▶ burstgpt_timeseries.csv / train_set.csv / warmup_set.csv
        │
        ├── scenario_extraction.py ──▶ data/scenarios/*.csv (실측 패턴)
        │
        └── models
             ├─ lstm_model.py  (Predictive, 오프라인 학습)
             ├─ linucb_agent.py (Reactive, 온라인 적응)
             └─ baseline.py     (Static P95)

Simulator (src/simulator.py)
        │
        └── Evaluation (src/evaluation.py)
                 ├─ experiments/run_all_scenarios.py
                 ├─ experiments/statistical_analysis.py
                 └─ plots/
```

---

## 2. 모듈별 개요

### 2.1 `src/data_pipeline.py`
- **역할**: BurstGPT CSV 로드 → 품질 검증 → 1초 단위 시계열 생성 → Train/Warmup/Test 분할.
- **핵심 함수**
  - `load_and_validate_burstgpt(Path) -> DataFrame`: Timestamp 단조 증가, 결측/토큰 무결성 검증.
  - `create_timeseries(DataFrame) -> DataFrame`: RPS, p99 latency, error_rate, cpu_percent 집계.
  - `add_multi_resolution_features(timeseries, raw, resolutions)`: 1s/100ms/10ms 등 서브초 통계 추가.
  - `PipelineConfig`: `multi_resolutions`, `enable_tfdv`, `tfdv_*_path` 등 파이프라인 옵션 정의.
  - `run_pipeline(config) -> Dict[str, Path]`: CSV(+TFDV 통계) 저장.

### 2.2 `src/scenario_extraction.py`
- **역할**: BurstGPT 실측 로그에서 네 가지 시나리오 추출.
  - `extract_periodic_conversation`, `extract_api_burst`, `extract_gradual_drift`, `extract_failure_spike`.
  - `ScenarioExtractionConfig`: 시간 구간, burst 배수, failure 임계값 등 설정.
  - `extract_burstgpt_scenarios(raw) -> Dict[str, ScenarioData]`.
- **스크립트**: `scripts/prepare_scenarios.py --data-dir data --output-dir data/scenarios`.

### 2.3 `src/lstm_model.py`
- **역할**: 예측형 Rate Limiter.
  - `LSTMPredictor(window_size=60, horizon=60, hidden_units=(64,32), dropout, learning_rate)`.
  - `fit(train_df, epochs, patience, samples_per_epoch, sampler)` – StratifiedRandomSequenceDataset로 시나리오 균형 학습.
  - `predict(context_df)` – 최근 window 기반 60초 수요 예측.
  - `save/load` – PyTorch state + scaler 저장.

### 2.4 `src/linucb_agent.py`
- **역할**: 컨텍스트 밴딧 기반 반응형 Rate Limiter.
  - 컨텍스트 예시: `[rps, error_rate, cpu_percent, rps_delta_5s, rps_std_30s, time_of_day_sin/cos]`.
  - `LinUCBAgent(action_space, alpha, alpha_decay=True)` – 계층형 액션 공간, α 감쇠 스케줄 지원.
  - `warmup(warmup_df)` – 초기 10% 데이터로 A/b 업데이트 및 탐험 시작.
  - `select_action(context)`, `update(context, action, reward)`.
  - `save/load` – JSON 기반 파라미터 직렬화.

### 2.5 `src/baseline.py`
- **역할**: Static P95 Rate Limiter 생성 (`StaticRateLimiter.from_data(train_df)`).

### 2.6 `src/simulator.py`
- **역할**: 오프라인 시뮬레이션.
  - `run_simulation(model, test_df, scenario, seed)` – 초단위로 throttle 적용 및 요청 수락 여부 계산.
  - 출력: `SimulationResult` (success_rate, p99_latency, stability_score, adaptation_time 등 + 상세 DataFrame).
  - 가드레일: LinUCB tracking lag, failure spike 복구 시간 계산.

### 2.7 `src/evaluation.py`
- **역할**: 메트릭 계산 및 통계 검정.
  - `compute_metrics(detailed_results, scenario)` – 시나리오별 핵심 지표 집계.
  - `run_statistical_tests(model1_results, model2_results)` – paired t-test, 효과크기 계산.
  - `generate_report(results_dict)` – 마크다운 요약 생성.

### 2.8 `experiments/` 스크립트
- `run_all_scenarios.py`
  - Train/Warmup/Test 로드 → 모델 준비 → 4 시나리오 × 3 모델 × seeds 실행.
  - 주요 CLI:
    - `--scenario-dir`: `scripts/prepare_scenarios.py` 산출물 사용.
    - `--linucb-context-keys`, `--linucb-decay-tau`, `--linucb-min-alpha`, `--disable-alpha-decay`: LinUCB 컨텍스트/탐험률 제어.
    - `--lstm-stratified`, `--samples-per-epoch`: LSTM 균형 샘플링 및 학습 스케줄 조정.
    - `--synthetic-only`: 실측 시나리오 없이 synthetic 기본값으로 회귀 테스트.
- `statistical_analysis.py`: 결과 JSON/CSV 로딩, 메트릭 요약, 유의성 검정, 리포트 저장.
- `visualization.py`: 성공률, p99, 안정도 등 그래프 생성.

### 2.9 `scripts/`
- `run_pipeline.py`: 파이프라인 실행 + `--with-tfdv` 옵션으로 TFDV 산출물 생성.
- `prepare_scenarios.py`: 실측 시나리오 CSV/메타데이터 생성.

---

## 3. 모델 비교 프레임워크 요약

1. **데이터 준비**
   - `run_pipeline.py` → `burstgpt_timeseries.csv`, `train_set.csv`, `warmup_set.csv`.
   - `prepare_scenarios.py` → `data/scenarios/{periodic,burst,drift,failure}.csv`.
2. **모델 학습**
   - LSTM: Train 70% (Stratified 샘플링, EarlyStopping).
   - LinUCB: Warmup 10% (확장 컨텍스트, α 감쇠).
3. **평가**
   - Simulator가 각 시나리오 CSV를 순회하며 Static/LSTM/LinUCB 결과 저장.
   - Evaluation 모듈이 메트릭/통계 분석 및 보고서 작성.
4. **결과 관리**
   - `results_<tag>/`, `plots/<tag>/`, `artifacts/<tag>/`로 버전별 산출물 분리 (`docs/EXPERIMENT_VERSIONS.md` 참조).
