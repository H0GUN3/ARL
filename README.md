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
cat CLAUDE.md

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
python src/data_pipeline.py
```

### 4단계: 모델 구현 및 학습
```bash
# 각 모듈은 @docs/API_DESIGN.md를 참고하여 Claude Code로 구현
# 구현 순서: data_pipeline → lstm_model → linucb_agent → simulator → evaluation
```

### 5단계: 실험 실행
```bash
python experiments/run_all_scenarios.py
```

### 6단계: 분석 및 시각화
```bash
python experiments/statistical_analysis.py
python experiments/visualization.py
```

---

## 폴더 구조

```
limiting/
├── CLAUDE.md                      # 🎯 스펙 주도 개발 가이드 (여기서 시작!)
├── docs/                          # 📋 명세 및 설계
│   ├── SPEC.md                    # 프로젝트 명세
│   ├── API_DESIGN.md              # API 설계
│   ├── IMPLEMENTATION_GUIDE.md     # 구현 절차
│   ├── TESTING_STRATEGY.md        # 테스트 전략
│   ├── AI_INTEGRATION.md          # Claude Code 사용법
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
4. **Claude Code로 구현** (@docs/AI_INTEGRATION.md 참고)
5. **테스트** (@docs/TESTING_STRATEGY.md)

**중요**: Claude Code 사용 시, 항상 @docs/*.md 파일을 참조합니다!

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
| CLAUDE.md | 모두 | SDD 워크플로우 |
| @docs/SPEC.md | 개발자 | 무엇을 만들 것인가 |
| @docs/API_DESIGN.md | 개발자 | 어떻게 만들 것인가 |
| @docs/IMPLEMENTATION_GUIDE.md | 개발자 | 단계별 구현 |
| @docs/AI_INTEGRATION.md | Claude Code 사용자 | AI와 함께 개발하는 법 |
| @docs/TESTING_STRATEGY.md | QA | 테스트 계획 |

---

## 프로젝트 상태

- [x] 명세 작성 (SPEC.md)
- [x] API 설계 (API_DESIGN.md)
- [x] 구현 가이드 (IMPLEMENTATION_GUIDE.md)
- [x] 폴더 구조 생성
- [ ] src/ 모듈 구현 (진행 중)
- [ ] tests/ 테스트 작성 (대기)
- [ ] experiments/ 실험 실행 (대기)
- [ ] results/ 분석 완료 (대기)

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

Day 3-4: 실험
  ├─ 60회 병렬 실행
  ├─ 결과 저장
  └─ 통계 분석

Day 5-7: 논문 작성
  ├─ 시각화
  ├─ 결과 정리
  └─ 논문 완성
```

---

## 파일별 책임

| 파일 | 담당 | 상태 |
|------|------|------|
| src/data_pipeline.py | Claude Code | [ ] 구현 예정 |
| src/lstm_model.py | Claude Code | [ ] 구현 예정 |
| src/linucb_agent.py | Claude Code | [ ] 구현 예정 |
| src/simulator.py | Claude Code | [ ] 구현 예정 |
| src/evaluation.py | Claude Code | [ ] 구현 예정 |
| tests/*.py | Claude Code | [ ] 구현 예정 |
| experiments/*.py | Claude Code | [ ] 구현 예정 |

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

프로젝트 관련 질문: CLAUDE.md 참조

