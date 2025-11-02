# 실험 버전 히스토리

| 버전 ID | 산출물 폴더 | 실행 시점 | 핵심 설정 | 주요 메트릭 요약 |
|---------|-------------|-----------|-----------|------------------|
| `v1` | `results_full/` | Phase 5 1차 실행 | Spec 스케일 재생성 데이터 (RPS 400–4000), LSTM 50 epoch·samples_per_epoch=200k, LinUCB α=0.25 고정, Static P95 ≈520 | 4 시나리오 × 3 모델 × 10 seeds = 120회. 예측형(LSTM)이 전반 우위이나 Gradual/Spike에서 MAE 개선 여지 남음. |
| `v1.1` | `results_full_rerun/` | Phase 5 재실험 (202k 샘플) | 데이터 재샘플링(epochs당 202k 시퀀스)으로 LSTM 학습 재수행, 시나리오 및 정책 동일 | LSTM Gradual 성공률 0.493, Normal 0.575 등 일부 개선. 재샘플링 효과 분석 목적. |
| `v1.2` | `results_full_fulltrain/` | Phase 5 최종 (전체 학습) | Train 70% 전체 샘플 random sampling, 동일 모델 설정 유지, 결과/문서 갱신 | LSTM 성공률: Gradual 0.632, Normal 0.9999, Periodic 0.734, Spike 0.698. LinUCB/Static와의 최종 비교 값으로 보고서 및 README 반영. |
| `v2` | `results_v2/` | Phase 6 파일럿 (2025-XX-XX) | 실측 기반 시나리오 추출(`scripts/prepare_scenarios.py`) + synthetic fallback, 파이프라인 스케일링(`run_pipeline.py --apply-scaling`), `run_all_scenarios.py --scenario-dir data/scenarios --lstm-stratified --seeds 0 1 2` | 시나리오별 성능 격차 확보: Burst/Drift에서 LinUCB 성공률 0.74/0.77, Static 0.42/0.31, LSTM 0.50/0.43. Failure에서는 LinUCB 1.0 vs Static 0.60. 실측 패턴 부족 시 Spec 기반 강도 보정으로 공정 비교 가능성 확인. |

## 참조 로그
- `AGENTS.md` 진행 로그(Phase 5 항목)  
- `docs/REPORT_DRAFT.md` – 각 버전의 설정·결과 서술  
- `results_full_fulltrain/statistical_report.md` – 최종 통계 분석

## 버전 관리 가이드
1. 새로운 실험을 실행할 때는 `results_<tag>/` 폴더를 생성하고, 동일한 태그를 `plots/<tag>/`, `artifacts/<tag>/` 등에 맞춰 사용합니다.  
2. 실험 메타데이터(스크립트 커맨드, 주요 하이퍼파라미터, 데이터 풀)와 날짜를 `docs/EXPERIMENT_VERSIONS.md`에 추가합니다.  
3. 중간 산출물(통계 리포트, 요약 CSV)은 버전 폴더 하위에 보관하고, README/보고서에는 최신 안정 버전만 반영합니다.
