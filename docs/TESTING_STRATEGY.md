# 테스트 및 검증 전략 (Testing Strategy)

## 0. 최신 스냅샷
- `v1.2` 결과 (`results_full_fulltrain/summary_metrics.csv`)는 Phase 5 기준 baseline 값이며, Phase 6 실측 시나리오 실험 후 갱신 예정이다.
- 시나리오별 승자 가설:
  - Periodic → LSTM
  - Burst / Failure → LinUCB
  - Drift → LSTM(초반) + LinUCB(후반)
- 통계 검정은 paired t-test + Cohen’s d를 사용해 전략 간 효과 크기를 보고한다.

## 1. 단위 테스트
- `tests/test_data_pipeline.py`: CSV 로드·시계열 집계·다중해상도 특성·파이프라인 저장 검증.
- `tests/test_scenario_generator.py`: 기존 synthetic 시나리오 회귀 확인.
- `tests/test_scenario_extraction` (추가 예정): 실측 시나리오 추출 로직 검증.
- `tests/test_lstm_model.py`: 학습/예측/저장·로드 동작, 무작위 seed 재현성.
- `tests/test_linucb_agent.py`: warmup, 업데이트, save/load.
- `tests/test_simulator.py`: 모델별 시뮬레이션 루프, 메트릭 계산 흐름.
- `tests/test_evaluation.py`: 메트릭 집계·통계 함수.
- `tests/test_statistical_analysis.py`, `tests/test_visualization.py`: 결과 요약 및 플롯 생성.

## 2. 통합 테스트 & 회귀
- `pytest` 전체 통과를 CI 선행 조건으로 유지 (`pytest` 실행 시 27개 테스트).
- 새로운 Phase 실험(v2) 결과는 `results_v2/`에 저장하고, `docs/EXPERIMENT_VERSIONS.md`에 메트릭을 기록한 후 baseline과 비교.
- 시나리오 CSV 변경 시 regression 플롯(성공률, p99, 안정도)을 비교하여 드리프트 여부 확인.

## 3. 통계 검증 절차
1. 시나리오별 각 모델의 성공률/지연/안정도 값 추출.
2. `run_statistical_tests`로 paired t-test, Cohen’s d, p-value 분석.
3. 리포트(`results/<tag>/statistical_report.md`)에 표/그래프/해석 기록.
4. 효과 크기가 medium 이상(>|0.5|)이면 논문 본문에 주요 결과로 포함.

## 4. Phase 6 점검 항목
- **Phase 1**: 기존 값 재현 (차이 미미함을 확인).
- **Phase 2**: LinUCB 컨텍스트·탐험률 개선 후 spike 시나리오에서 성능 향상 확인.
- **Phase 3**: LSTM Stratified 샘플링 후 drift/periodic 시나리오 개선 확인.
- **Phase 4**: Stress Mix/Failure 시나리오에서 가드레일 위반 횟수 비교.

## 5. 로그 & 아티팩트
- `artifacts/tfdv/`: TFDV 통계/스키마/이상치 결과.
- `artifacts/<tag>/`: 실험별 추가 검증 자료(예: MLflow 로그, 학습 곡선)를 저장할 예정.

> Phase 6 실험이 완료되면 본 문서를 다시 갱신하여 신규 결과와 검증 절차를 반영한다.
