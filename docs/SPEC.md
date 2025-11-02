# 프로젝트 명세 (Project Specification)

## 1. 프로젝트 개요

### 1.1 목표
현실 LLM 서비스 트래픽(BurstGPT)에서 **Predictive 전략(LSTM)**과 **Reactive 전략(LinUCB)** 을 동일한 데이터·환경·메트릭으로 공정 비교해, 트래픽 패턴별 최적 Rate Limiting 정책을 제시한다.

### 1.2 데이터셋
- **이름**: BurstGPT v1.1
- **크기**: 5.29M traces, 121일 (2023년)
- **출처**: Azure OpenAI API (공개)
- **포함 정보**: timestamp, model, request_tokens, response_tokens, total_tokens, log_type

### 1.3 비교 대상
| 전략 | 유형 | 주요 특성 | 의사결정 주기 |
|------|------|---------|------------|
| **Static** | 기준선 | Train P95임계값 | N/A |
| **LinUCB** | Reactive | 컨텍스트 기반 온라인 적응 | 1초 |
| **LSTM** | Predictive | 시계열 기반 선제 예측 | 60초 |

### 1.4 기간
- **실험 기간**: 7일
- **구현**: Day 1-2
- **실험 실행**: Day 3-4
- **논문 작성**: Day 5-7

---

## 2. 데이터 명세

### 2.1 입력 데이터
**원본 파일**: `BurstGPT_1.csv`, `BurstGPT_2.csv`

**시계열 변환 결과(`burstgpt_timeseries.csv`) 컬럼**:

| 컬럼 | 타입 | 설명 |
|------|------|------|
| second | int | Unix timestamp (1초 단위) |
| rps | float | 초당 요청 수 |
| p99_latency | float | 응답 토큰 기반 99th percentile latency (ms) |
| error_rate | float | 실패 요청 비율 (0-1) |
| cpu_percent | float | 토큰 throughput 기반 CPU 사용률 (0-100) |

### 2.2 데이터 분할
Train: Day 1-84 (70%) – LSTM 학습  
Warmup: Day 85-96 (10%) – LinUCB 초기 적응  
Test: Day 97-121 (20%) – 실측 시나리오 추출

### 2.3 시나리오 정의
**4개 실측 시나리오 (BurstGPT 기반)**

| 시나리오 | 추출 대상 | 주요 특징 | 비교 목적 |
|----------|-----------|-----------|-----------|
| Conversation-Periodic | Conversation log (ChatGPT) 평일 근무시간 | 일일/주간 주기, CV≈0.3 | 패턴 학습 능력(LSTM) 검증 |
| API-Burst | API log (GPT-4) 급증 구간 | 8~12배 급증, failure spike 23% | 즉각 적응(LinUCB) 능력 검증 |
| Gradual-Drift | 사용자 증가 구간 | 2~3배 성장, 분포 변동 | 분포 변화 대응(LSTM vs LinUCB) |
| Failure-Spike | 실패율 집중 구간 | 정상 RPS → 에러율 급증 | 실패 복구 속도(LinUCB 가드레일) |

시나리오 생성은 `scripts/prepare_scenarios.py`로 BurstGPT 원본에서 직접 추출한다.

---

## 3. 모델 명세

### 3.1 Static (기준선)
```python
class StaticRateLimiter:
    threshold = percentile_95(train_rps)  # 고정값
    action: reject if rps > threshold
```

**입출력**:
- Input: current_rps
- Output: allow (1) or reject (0)

### 3.2 LSTM (Predictive)
```python
class LSTMPredictor:
    input: past_60_sec_features  # [RPS, P99, Error, CPU] × 60초
    output: predict_next_60_sec_rps
    architecture:
      LSTM(64) → LSTM(32) → Dense(16) → Dense(horizon)
    learning: offline supervised (70% data)
    decision: if predicted_rps > threshold → pre-limit
```

**하이퍼파라미터 (고정)**:
- window_size: 60초
- prediction_horizon: 60초
- dropout: 0.2
- batch_size: 32
- epochs: 50 (Early stopping, patience=5)
- optimizer: Adam (lr=0.001)
- loss: MSE
- 입력 피처: `[rps, p99_latency, error_rate, cpu_percent]`

### 3.3 LinUCB (Reactive)
```python
class LinUCBAgent:
    input: current_context (rps, error_rate, cpu)
    output: action (throttle limit)
    learning: online RL (10% warmup)
    decision: every 1 second
```

**하이퍼파라미터**:
- action_space: 50 (500-5000 RPS, 100 단위)
- exploration_rate: sqrt(ln(t) / t)
- update_freq: 1초

---

## 4. 평가 메트릭

### 4.1 P99 Latency
```
계산: response_tokens × 30ms/token
신뢰도: ⭐⭐⭐ (상대 비교만 유효)
범위: 0-2000ms
목표: 낮을수록 좋음
```

### 4.2 Success Rate
```
계산: (total_requests - rejected) / total_requests
신뢰도: ⭐⭐⭐⭐⭐ (100%)
범위: 0-100%
목표: 높을수록 좋음
```

### 4.3 Stability Score
```
계산: % 시간동안 P99 < 150ms (SLA 준수율)
신뢰도: ⭐⭐⭐⭐
범위: 0-100%
목표: 높을수록 좋음
```

### 4.4 Adaptation Time (Spike 시나리오만)
```
계산: Spike 발생 후 정상 범위 복구 시간
신뢰도: ⭐⭐⭐⭐
범위: 0-300초
목표: 낮을수록 좋음
```

### 4.5 Predictive Accuracy (Gradual 시나리오)
```
계산: |예측 RPS - 실제 RPS|의 MAE (LSTM 전용)
신뢰도: ⭐⭐⭐⭐
범위: 0-500 RPS
목표: 낮을수록 좋음
```

### 4.6 Tracking Lag (Gradual 시나리오, LinUCB)
```
계산: throttle_limit < 실제 RPS인 초(second) 누적
신뢰도: ⭐⭐⭐
범위: 0-지속시간
목표: 낮을수록 좋음
```

### 4.7 Pattern Recognition (Periodic 시나리오)
```
계산: RPS 시계열의 자기상관 (lag = period/2)
신뢰도: ⭐⭐⭐
범위: -1~1
목표: 1에 가까울수록 강한 패턴 인식
```

### 4.8 Proactive Adjustment (Periodic 시나리오)
```
계산: 전환 시점 전/후 ±30초 error rate 차이 (후 - 전)
신뢰도: ⭐⭐⭐
범위: -1~1
목표: 음수(전환 전에 미리 대비)
```

---

## 5. 실험 설계

### 5.1 반복 실행
```
각 시나리오 × 모델 조합마다:
- 10회 반복 (seed 0-9)
- 5개 시간 구간 각각에서 평가
- 총: 4 시나리오 × 3 모델 × 10 seeds = 120회
```

### 5.2 통계 검증
```
메트릭: Paired t-test
신뢰도: 95% (α = 0.05)
가설: H0: LinUCB = LSTM
```

### 5.3 Convergence 확인
**LSTM**:
- Train loss curve 모니터링
- 70% 데이터에서 수렴 확인

**LinUCB**:
- Regret curve 모니터링
- 10% 데이터에서 수렴 확인

---

## 6. 출력 명세

### 6.1 결과 파일
```
results/
├── {Model}_{Scenario}_{Seed}.json
├── {Model}_{Scenario}_{Seed}_details.csv
└── statistical_report.md  # 통합 통계 요약
```

### 6.2 시각화
```
plots/
├── comparison_p99_boxplot.png
├── success_rate_barplot.png
└── ...
```

### 6.3 논문 (4-6페이지)
- Introduction: 연구 배경 및 기여
- Related Work: 기존 연구 분류
- Methodology: BurstGPT, 모델, 평가 방법
- Experiments: 결과 및 분석
- Discussion & Conclusion: 주요 발견 및 제한사항

---

## 7. 공정성 보장 기준

### 7.1 "공정한 비교" 5가지 조건
1. ✅ **동일 입력 데이터**: 모든 모델이 동일한 RPS, P99, Error 관찰
2. ✅ **동일 평가 메트릭**: 중립적 지표 (P99, Success Rate)
3. ✅ **수렴 보장**: Training/Regret curve로 수렴 확인
4. ✅ **Time leakage 방지**: 미래 정보 사용 금지
5. ✅ **동일 실험 환경**: 동일 시뮬레이터, 동일 시간 구간

### 7.2 제한사항 명시
- ⚠️ P99는 추정값 (절대값 신뢰도 제한, 상대 비교만 유효)
- ⚠️ 4개 시나리오이지만 축약 데이터셋(샘플) 기반
- ⚠️ n=10 (통계적 완전성 한계)
- → Limitations 섹션에 명확히 표기

---

## 8. 의존성 및 환경

### 8.1 언어 및 라이브러리
- Python 3.9+
- PyTorch 2.x (LSTM)
- NumPy, Pandas (데이터 처리)
- Scipy (통계)
- Matplotlib, Seaborn (시각화)

### 8.2 하드웨어
- GPU: RTX A6000 (권장)
- 예상 시간: 45시간 (병렬 실행 시 2일)

### 8.3 재현성
- Random seed 고정: 42
- 모든 하이퍼파라미터 공개
- 코드 및 데이터 GitHub 공개

---

## 9. 변경 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|---------|
| v1.0 | 2025-10-29 | 초안 작성 |
| v1.1 | 2025-10-29 | AI 피드백 반영 (60회 실험, 2개 시나리오) |
| v1.2 | 2025-10-30 | Gradual·Periodic 시나리오 및 신규 메트릭 반영 |
| v2.0 | TBD | 최종 확정 |

---

## 10. 승인 및 체크리스트

- [x] 명세 검토 완료
- [x] 데이터 가용성 확인
- [x] 모델 구현 가능성 검증
- [x] 일정 확인 (7일 내)
- [x] 의존성 설치 완료

---

## 11. 실험 결과 해석 (요약)

120회 반복 실험(Train 70 % 전체 사용, `samples_per_epoch=200,000`)에 대한 핵심 해석은 다음과 같다.

- **Gradual (점진적 증가)**: LSTM 평균 성공률 0.632로 Static(0.463)·LinUCB(0.455)을 크게 앞섰다. ramp-up 구간에서도 허용률 0.6 이상을 유지하며 안정성 점수도 0.40 수준까지 끌어올렸다(MAE ≈ 1.5k RPS).
- **Normal (정상 패턴)**: LSTM이 0.9999에 가까운 성공률로 대부분의 요청을 수용했다(Static 0.634, LinUCB 0.610). paired t-test 결과 p<1e-6으로 LSTM이 LinUCB 대비 통계적으로 우월하다.
- **Periodic (주기 패턴)**: LSTM 0.734 > Static 0.491 > LinUCB 0.473. 예측 기반 전략이 전환 지점을 선제 처리하며, error 전환 차이(±30초)가 음수로 개선됐다.
- **Spike (급증)**: LSTM 0.698, Static 0.609, LinUCB 0.605. 적응 시간은 LinUCB/Static이 89초 수준에 머무르지만, LSTM은 spike 구간에서도 허용률을 70 % 가까이 끌어올렸다.

결론적으로 전체 학습 데이터를 사용하면 LSTM이 4개 시나리오 전반에서 명확한 우위를 보인다. 남은 과제는 (1) Gradual/Periodic에서의 MAE 추가 감소, (2) LinUCB 탐색 정책 개선, (3) Static 임계값의 동적 보조 전략 설계다.
