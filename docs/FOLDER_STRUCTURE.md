# 폴더 구조 정의 (Folder Structure)

```
.
├── AGENTS.md
├── README.md
├── artifacts/
│   ├── README.md
│   └── tfdv/ (stats.pbtxt, schema.pbtxt, anomalies.pbtxt 등)
├── data/
│   ├── BurstGPT_*.csv        # 원본 (외부 다운로드)
│   ├── burstgpt_timeseries.csv
│   ├── train_set.csv / warmup_set.csv / test_set.csv
│   └── scenarios/            # scripts/prepare_scenarios.py 출력
├── docs/
│   ├── SPEC.md
│   ├── API_DESIGN.md
│   ├── IMPLEMENTATION_GUIDE.md
│   ├── TESTING_STRATEGY.md
│   ├── EXPERIMENT_REDESIGN.md
│   ├── EXPERIMENT_VERSIONS.md
│   └── PHASE.md
├── experiments/
│   ├── run_all_scenarios.py
│   ├── statistical_analysis.py
│   └── visualization.py
├── plots/
│   ├── full_fulltrain/
│   └── (버전별 시각화 산출물)
├── results/
│   └── (임시 실험 결과; 버전별 폴더는 results_* 사용)
├── results_full/
├── results_full_rerun/
├── results_full_fulltrain/
├── scripts/
│   ├── run_pipeline.py
│   └── prepare_scenarios.py
├── src/
│   ├── __init__.py
│   ├── baseline.py
│   ├── data_pipeline.py
│   ├── evaluation.py
│   ├── linucb_agent.py
│   ├── lstm_model.py
│   ├── scenario_extraction.py
│   ├── scenario_generator.py
│   └── simulator.py
├── tests/                     # pytest 기반 단위/통합 테스트
│   ├── test_data_pipeline.py
│   ├── test_evaluation.py
│   ├── test_linucb_agent.py
│   ├── test_lstm_model.py
│   ├── test_scenario_generator.py
│   ├── test_simulator.py
│   ├── test_statistical_analysis.py
│   └── test_visualization.py
├── requirements.txt
└── venv/ (로컬 가상환경)
```

## 메모
- 실험 버전별 산출물은 `results_<tag>/`, `plots/<tag>/`, `artifacts/<tag>/`에 저장하고 `docs/EXPERIMENT_VERSIONS.md`에 기록합니다.
- 실측 시나리오 준비는 `scripts/prepare_scenarios.py`를 통한 CSV 생성 → `data/scenarios/`.
- Phase 6 이후 추가되는 파일/폴더는 이 문서를 최신 상태로 유지해 추적합니다.
