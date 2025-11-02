# API Rate Limiting 전략 비교 연구

## 프로젝트 개요

**목표**: LSTM (Predictive) vs LinUCB (Reactive) 기반 Rate Limiting 전략을 동일 환경에서 공정하게 비교하여 모델 선택 가이드라인 제시

**데이터**: BurstGPT v1.1 (5.29M traces, 121일)
**기간**: 7일 (구현 2일 + 실험 2일 + 분석 3일)
**방법론**: Spec-Driven Development (SDD)

---

## 빠른 시작

### 1단계: 문서 읽기
```bash
# 스펙 주도 개발 가이드
cat AGENTS.md

# 프로젝트 명세
cat docs/SPEC.md

# API 설계
cat docs/API_DESIGN.md
```

### 2단계: 환경 설정
```bash
pip install -r requirements.txt
```

### 3단계: 데이터 준비
```bash
# BurstGPT v1.1 다운로드
# https://github.com/HPMLL/BurstGPT/releases/tag/v1.1
# → data/burstgpt_v1.1.csv로 저장

# 데이터 전처리
# 기본 파이프라인 실행
python src/data_pipeline.py

# (선택) TFDV 통계/스키마/이상치 리포트 생성
# pip install tensorflow-data-validation 후 실행
python scripts/run_pipeline.py --with-tfdv

# (선택) BurstGPT 기반 시나리오 추출
python scripts/prepare_scenarios.py --data-dir data --output-dir data/scenarios
```

### 4단계: 모델 구현 및 학습
```bash
# 각 모듈은 @docs/API_DESIGN.md를 참고하여 Codex로 구현
# 구현 순서: data_pipeline → lstm_model → linucb_agent → simulator → evaluation
```

### 5단계: 실험 실행
```bash
python experiments/run_all_scenarios.py \
  --scenario-dir data/scenarios \
  --linucb-context-keys rps,error_rate,cpu_percent,rps_delta_5s,rps_std_30s,time_of_day_sin,time_of_day_cos \
  --lstm-stratified \
  --seeds 0 1 2
```

- `--scenario-dir`: `scripts/prepare_scenarios.py`가 생성한 BurstGPT 실측 시나리오를 로드합니다. (없으면 synthetic 시나리오 자동 생성)
- `--linucb-context-keys`: LinUCB 컨텍스트 피처 목록을 지정합니다. 기본은 7개 확장 피처(급증 탐지 + 시간 인코딩)입니다.
- `--lstm-stratified`: 학습 샘플을 시나리오별 균등 분포로 뽑아 드리프트/버스트 처리 능력을 키웁니다.
- `--seeds`: 시나리오 × 모델 반복 실행을 위한 시드 리스트입니다.
- `--synthetic-only`를 지정하면 실측 CSV 없이도 빠른 회귀 테스트를 수행할 수 있습니다.

### 6단계: 분석 및 시각화
```bash
python experiments/statistical_analysis.py
python experiments/visualization.py
```

---

## 폴더 구조

```
limiting/
├── AGENTS.md                      # 🎯 스펙 주도 개발 가이드 (여기서 시작!)
├── docs/                          # 📋 명세 및 설계
│   ├── SPEC.md                    # 프로젝트 명세
│   ├── API_DESIGN.md              # API 설계
│   ├── IMPLEMENTATION_GUIDE.md     # 구현 절차
│   ├── TESTING_STRATEGY.md        # 테스트 전략
│   ├── AI_INTEGRATION.md          # Codex 사용법
│   └── FOLDER_STRUCTURE.md        # 폴더 구조
├── src/                           # 🔨 소스 코드
├── tests/                         # ✅ 테스트
├── experiments/                   # 🧪 실험 스크립트
├── data/                          # 📊 데이터
├── models/                        # 💾 학습된 모델
├── results/                       # 📈 실험 결과
└── plots/                         # 📉 시각화
```

---

## 스펙 주도 개발 원칙

**모든 개발은 다음 순서로 진행됩니다**:

1. **명세 작성** (@docs/SPEC.md)
2. **API 설계** (@docs/API_DESIGN.md)
3. **구현 가이드** (@docs/IMPLEMENTATION_GUIDE.md)
4. **Codex로 구현** (@docs/AI_INTEGRATION.md 참고)
5. **테스트** (@docs/TESTING_STRATEGY.md)

**중요**: Codex 사용 시, 항상 @docs/*.md 파일을 참조합니다!

```
❌ 나쁜 예:
"LSTM 만들어줘"

✅ 좋은 예:
"@docs/API_DESIGN.md 섹션 2.2의 LSTMPredictor를 참고해서
 src/lstm_model.py를 구현해줘"
```

---

## 핵심 문서 한눈에

| 문서 | 대상자 | 목적 |
|------|--------|------|
| AGENTS.md | 모두 | SDD 워크플로우 |
| @docs/SPEC.md | 개발자 | 무엇을 만들 것인가 |
| @docs/API_DESIGN.md | 개발자 | 어떻게 만들 것인가 |
| @docs/IMPLEMENTATION_GUIDE.md | 개발자 | 단계별 구현 |
| @docs/AI_INTEGRATION.md | Codex 사용자 | AI와 함께 개발하는 법 |
| @docs/TESTING_STRATEGY.md | QA | 테스트 계획 |

---

## 프로젝트 상태

- [x] 명세 작성 (SPEC.md)
- [x] API 설계 (API_DESIGN.md)
- [x] 구현 가이드 (IMPLEMENTATION_GUIDE.md)
- [x] 폴더 구조 생성
- [x] src/ 모듈 구현 (데이터 파이프라인/모델/시나리오/시뮬레이터)
- [x] tests/ 테스트 작성 (단위·통합·시나리오)
- [x] experiments/ 실험 실행 스크립트 구성
- [x] results/ 분석 보고서 작성 (`results_full/statistical_report.md`, 120-run 실험 반영)

---

## 실험 결과 요약 (Success Rate, seed 0-9 평균)

| Scenario  | LSTM | LinUCB | Static |
|-----------|------|--------|--------|
| Gradual   | 0.632 | 0.455 | 0.463 |
| Normal    | 1.000 | 0.610 | 0.634 |
| Periodic  | 0.734 | 0.473 | 0.491 |
| Spike     | 0.698 | 0.605 | 0.609 |

- 전체 지표/통계: `results_full_fulltrain/summary_metrics.csv`, `results_full_fulltrain/statistical_report.md`
- 시각화: `plots/full_fulltrain/comparison_p99_boxplot.png`, `plots/full_fulltrain/success_rate_barplot.png`, `plots/full_fulltrain/stability_score_barplot.png`
- 요약 보고: `docs/REPORT_DRAFT.md`

---

## 7일 일정

```
Day 1-2: 구현
  ├─ data_pipeline.py
  ├─ lstm_model.py
  ├─ linucb_agent.py
  ├─ simulator.py
  ├─ evaluation.py
  └─ 단위 테스트

Day 3-4: 실험 설계 & 시뮬레이션
  ├─ Normal/Spike/Gradual/Periodic 시나리오 생성
  ├─ LSTM/LinUCB/Static 준비
  ├─ 시뮬레이터 개선 및 검증

Day 5-6: 실험 실행
  ├─ 4 시나리오 × 3 모델 × 10 seeds (120 runs)
  ├─ 결과 저장 (`results/`, `plots/`)
  └─ 통계 분석/시각화 자동화

Day 7: 논문/보고서 초안
  ├─ 시각화 정리
  ├─ 핵심 결과/통계 요약
  └─ 결론 및 논의
```

---

## 파일별 책임

| 파일 | 담당 | 상태 |
|------|------|------|
| src/data_pipeline.py | Codex | [x] 상세 명세 반영 |
| src/lstm_model.py | Codex | [x] 다중 피처 LSTM 구현 |
| src/linucb_agent.py | Codex | [x] 워밍업/저장 기능 |
| src/scenario_generator.py | Codex | [x] Normal/Spike/Gradual/Periodic |
| src/simulator.py | Codex | [x] 추가 메트릭 포함 |
| src/evaluation.py | Codex | [x] 신규 메트릭 계산 |
| tests/*.py | Codex | [x] 단위·통합 테스트 |
| experiments/*.py | Codex | [x] 실행/통계/시각화 스크립트 |

---

## 의존성

```
Python 3.9+
├── torch (LSTM)
├── pandas (데이터)
├── numpy (수치)
├── scipy (통계)
├── scikit-learn (유틸)
├── matplotlib (시각화)
└── pytest (테스트)
```

**설치**:
```bash
pip install -r requirements.txt
```

---

## 참고 자료

- 검증.md - 방법론 검증 (기존 문서)
- AI문제.md - AI 피드백 (해결 완료)
- BurstGPT: https://github.com/HPMLL/BurstGPT

---

## 라이선스

MIT License

---

## 연락처

프로젝트 관련 질문: AGENTS.md 참조
