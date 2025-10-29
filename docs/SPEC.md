# 프로젝트 명세 (Project Specification)

## 1. 프로젝트 개요

### 1.1 목표
API Rate Limiting 분야에서 **Reactive 전략 (LinUCB)** 과 **Predictive 전략 (LSTM)** 을 동일 환경에서 공정하게 비교하여 모델 선택 가이드라인 제시

### 1.2 데이터셋
- **이름**: BurstGPT v1.1
- **크기**: 5.29M traces, 121일 (2023년)
- **출처**: Azure OpenAI API (공개)
- **포함 정보**: timestamp, model, request_tokens, response_tokens, total_tokens, log_type

### 1.3 비교 대상
| 전략 | 유형 | 주요 특성 | 의사결정 주기 |
|------|------|---------|------------|
| **Static** | 기준선 | 고정값 (P95 RPS) | N/A |
| **LinUCB** | Reactive | 온라인 적응 학습 | 1초 |
| **LSTM** | Predictive | 오프라인 예측 학습 | 60초 |

### 1.4 기간
- **실험 기간**: 7일
- **구현**: Day 1-2
- **실험 실행**: Day 3-4
- **논문 작성**: Day 5-7

---

## 2. 데이터 명세

### 2.1 입력 데이터
**파일**: burstgpt_timeseries.csv

| 컬럼 | 타입 | 설명 | 예시 |
|------|------|------|------|
| timestamp | datetime | UTC 시간 | 2023-01-01 00:00:00 |
| rps | float | Request Per Second | 1250.5 |
| p99_latency | float | 토큰 기반 추정 (ms) | 450.0 |
| error_rate | float | 실패율 (0-1) | 0.02 |
| model | string | 모델명 | gpt-4, gpt-3.5 |
| response_tokens | int | 평균 응답 토큰 | 15 |

### 2.2 데이터 분할
```
전체 (121일, 5.29M traces)
├─ Train: Day 1-84 (70%, 1.76M)
├─ Warmup: Day 85-96 (10%, 0.25M)
└─ Test: Day 97-121 (20%, 0.5M) × 5 시간 구간
    ├─ Test_1: Day 97-101
    ├─ Test_2: Day 102-106
    ├─ Test_3: Day 107-111
    ├─ Test_4: Day 112-116
    └─ Test_5: Day 117-121
```

### 2.3 시나리오 정의
**2개 시나리오만 (현실적 범위)**

#### 시나리오 1: Normal (정상 트래픽)
- RPS: 안정적 ± 10%
- 주기: 일일 패턴 반복
- 목표: 기본 성능 비교

#### 시나리오 2: Spike (급증)
- RPS: 5배 이상 급격한 증가 (< 10초)
- 지속: 최소 30초 이상
- 목표: 적응 능력 비교 (LinUCB 유리 예상)

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
    input: past_60_sec_rps  # 60초 과거 데이터
    output: predict_next_60_sec_rps
    learning: offline supervised (70% data)
    decision: if predicted_rps > threshold → pre-limit
```

**하이퍼파라미터**:
- window_size: 60초
- hidden_units: 128
- dropout: 0.2
- epochs: 50
- loss: MSE

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

---

## 5. 실험 설계

### 5.1 반복 실행
```
각 시나리오 × 모델 조합마다:
- 10회 반복 (seed 0-9)
- 5개 시간 구간 각각에서 평가
- 총: 2 시나리오 × 3 모델 × 10 seeds = 60회
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
├── scenario_1_normal_results.csv
│   ├── model, seed, time_window, p99_latency, success_rate, stability_score
├── scenario_2_spike_results.csv
│   ├── model, seed, time_window, p99_latency, success_rate, adaptation_time
└── statistical_summary.txt
    ├── t-test results, p-values, effect size
```

### 6.2 시각화
```
plots/
├── comparison_p99_boxplot.png
├── success_rate_barplot.png
├── stability_trend.png
├── adaptation_time_histogram.png
└── convergence_curves.png
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
- ⚠️ 2개 시나리오만 (proof-of-concept)
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
| v2.0 | TBD | 최종 확정 |

---

## 10. 승인 및 체크리스트

- [ ] 명세 검토 완료
- [ ] 데이터 가용성 확인
- [ ] 모델 구현 가능성 검증
- [ ] 일정 확인 (7일 내)
- [ ] 의존성 설치 완료

