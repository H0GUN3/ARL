# 개발 단계 및 체크리스트

Codex 작업을 일관되게 관리하기 위해 개발 순서를 다음과 같이 정의합니다.
각 단계가 끝날 때마다 체크박스를 갱신하고, 진행 상황은 `AGENTS.md`에
기록합니다.

## Phase 0. 환경 및 데이터 준비
- [x] `venv` 가상환경 생성 및 `requirements.txt` 설치
- [x] `data/` 폴더 정리 및 원본 데이터 무결성 확인
- [x] `scripts/eda.py` 실행 후 `scripts/eda_report.md` 검토

## Phase 1. 데이터 파이프라인 구축
- [x] BurstGPT 원본 로드 및 무결성 검증 (`load_and_validate_burstgpt`)
- [x] 1초 시계열 생성 (`create_timeseries`) 및 품질 검증 6항목 통과
- [x] 70/10/20 시간 순서 분할 (`split_dataset`)과 CSV 저장 (train/warmup/test)

## Phase 2. 모델 & 시나리오 준비
- [x] 시나리오 생성기 구현/검증 (normal, spike, gradual, periodic)
- [x] `src/lstm_model.py` 다중 피처 LSTM 학습 파이프라인 구현
- [x] `src/linucb_agent.py` LinUCB 워밍업/수렴 로직 구현

## Phase 3. 시뮬레이션 및 실험 환경
- [x] `src/simulator.py` 통합 테스트 통과 (추가 메트릭 포함)
- [x] `experiments/run_all_scenarios.py`로 4 시나리오 × 3 모델 실행 플로우 구성
- [x] 결과 저장 구조 (`results/`, `plots/`) 및 재현성 점검

## Phase 4. 평가 및 분석
- [x] `src/evaluation.py` 지표 계산 로직 구현 (`pytest tests/test_evaluation.py`)
- [x] `experiments/statistical_analysis.py` 결과 재현성 확인 (`pytest tests/test_statistical_analysis.py`)
- [x] `plots/` 시각화 산출물 검토 및 논문용 그래프 확정 (`experiments/visualization.py`, `pytest tests/test_visualization.py`)

## Phase 5. 문서화 및 최종 검토
- [x] `docs/TESTING_STRATEGY.md`와 테스트 결과 일치 여부 확인
- [x] 논문 초안/보고서에 실험 결과 반영 (`results_full_fulltrain/summary_metrics.csv`, `results_full_fulltrain/statistical_report.md`, LSTM 50 epoch + early stopping + random sampling)
- [x] `docs/PHASE.md` 체크리스트 및 `AGENTS.md` 진행 로그 최종 업데이트

## Phase 6. 실험 재설계 실행
- [ ] 데이터 파이프라인 보강: TFDV 통합, 다중 해상도(1초/100ms/10ms) 집계, request_id 기반 정합성 검증, BurstGPT 실측 시나리오 추출(`scripts/prepare_scenarios.py`)
- [ ] 모델 재튜닝: Adaptive Static baseline, LinUCB 탐험률 스케줄·가드레일, LSTM 균형 샘플링·하이퍼파라미터 탐색
- [ ] 시나리오/시뮬레이터 강화: 파라미터 랜덤화, Stress Mix 추가, 큐잉 기반 throttle 처리
- [ ] 통계 및 거버넌스: 파워/효과크기 보고, 다중 검정 보정, 실험 체크리스트/가드레일 확립
- [ ] 파일럿 재실험: 신규 파이프라인으로 전체 시나리오 + Stress Mix를 실행하고 결과/문서 업데이트
