# AI 도구 통합 가이드 (AI Integration Guide)

## 1. Codex와 SDD (Spec-Driven Development)

### 1.1 Codex 사용 워크플로우

```
명세 작성 (@docs/SPEC.md)
        ↓
API 설계 (@docs/API_DESIGN.md)
        ↓
구현 지시 작성 (명확한 지시)
        ↓
Codex 실행
  → 지시와 @docs/*.md 참조
  → 명세에 맞는 코드 생성
        ↓
코드 검증 (@docs/TESTING_STRATEGY.md)
        ↓
반복 수정 (필요시)
```

### 1.2 Codex에 지시할 때 원칙

✅ **DO (해야 할 것)**:
```
"@docs/API_DESIGN.md에서 LSTMPredictor.fit() 명세를 보고
data_pipeline.py의 aggregate_to_timeseries() 결과를 입력으로
LSTM 모델을 구현해줘.

조건:
- window_size: 60
- hidden_units: 128
- Loss 함수: MSE
- 수렴 확인: 마지막 10 epoch의 손실 차 < 0.01
"
```

❌ **DON'T (하지 말아야 할 것)**:
```
"LSTM 모델 만들어줘"  (너무 모호함)
"적절히 최적화해줘"  (명확하지 않음)
"기존 코드 수정해줘" (어느 부분인지 명시 필요)
```

---

## 2. 모듈별 Codex 사용법

### 2.1 data_pipeline.py 구현

**명세 참조**: @docs/API_DESIGN.md 섹션 2.1

**Codex 지시 예시**:
```
@docs/API_DESIGN.md의 data_pipeline.py 섹션을 보면서
src/data_pipeline.py를 구현해줘.

요구사항:
1. load_burstgpt(csv_path) - BurstGPT CSV 로드
   - 필수 컬럼: timestamp, request_tokens, response_tokens, model, log_type
   - 반환: DataFrame

2. aggregate_to_timeseries(df, interval=1) - 1초 단위 집계
   - rps = count(requests) / interval
   - p99_latency = quantile(response_tokens, 0.99) × 30
   - error_rate = count(response_tokens == 0) / count(all)

3. split_data(df) - 70/10/20 분할
   - Train: Day 1-84
   - Warmup: Day 85-96
   - Test: Day 97-121 (5등분)
   - Time leakage 방지 검증

4. save_timeseries(df, output_path) - CSV로 저장

테스트: tests/test_data_pipeline.py와 일치해야 함
```

### 2.2 LSTM 모델 구현

**명세 참조**: @docs/API_DESIGN.md 섹션 2.2, @docs/IMPLEMENTATION_GUIDE.md 섹션 3.1

**Codex 지시**:
```
@docs/API_DESIGN.md를 보면서 src/lstm_model.py를 구현해줘.

클래스: LSTMPredictor
속성:
- window_size: 60초
- hidden_units: 128
- dropout: 0.2
- device: cuda or cpu

메서드:
1. fit(train_data, epochs=50, batch_size=32, samples_per_epoch=200_000)
   - 입력: DataFrame with 'rps' column (1.76M 초)
   - 손실함수: MSE
   - 옵티마이저: Adam (lr=0.001)
   - 수렴 확인: 마지막 10 epoch 손실 차 < 0.01

2. predict(context)
   - 입력: DataFrame with 마지막 60초 데이터
   - 출력: 미래 60초 RPS 예측 (shape: (60,))
   - 정규화/역정규화 처리

3. save(path) / load(path)
   - torch.save() / torch.load() 사용

구현 세부:
- PyTorch LSTM 사용
- DataLoader로 배치 처리
- 진행률 바(tqdm) 표시
```

### 2.3 LinUCB Agent 구현

**명세 참조**: @docs/API_DESIGN.md 섹션 2.3

**Codex 지시**:
```
@docs/API_DESIGN.md를 참고하여 src/linucb_agent.py를 구현해줘.

클래스: LinUCBAgent
속성:
- action_space: [500, 600, ..., 5000] (50개)
- alpha: 0.25 (신뢰도 파라미터)
- A: dict of d×d 단위 행렬 (각 action마다)
- b: dict of d 영벡터 (각 action마다)
- d: 3 (context dimension)

메서드:
1. warmup(warmup_data)
   - 입력: DataFrame (0.25M 초, 10%)
   - 매 초마다: context 추출 → action 선택 → reward → 파라미터 업데이트
   - Regret curve 모니터링
   - 수렴 확인: 마지막 1000초의 regret 기울기 < threshold

2. select_action(context)
   - LinUCB 식: arm = argmax_a (θ_a^T x + α √(x^T A_a^{-1} x))
   - θ_a = A_a^{-1} b_a
   - 반환: action index (0-49)

3. update(context, action, reward)
   - A[action] += context @ context.T
   - b[action] += reward * context

4. save(path) / load(path)
   - JSON 형식으로 A, b 저장/로드
```

### 2.4 Simulator 구현

**명세 참조**: @docs/API_DESIGN.md 섹션 2.4

**Codex 지시**:
```
@docs/API_DESIGN.md를 참고하여 src/simulator.py를 구현해줘.

함수:
1. run_simulation(model, test_data, scenario, seed)
   - 입력: 학습된 모델, 평가 데이터 (0.5M 초), 시나리오, seed
   - 동작:
     for t in test_data:
       context = extract_context(rps, error_rate, cpu)
       action = model.predict(context)
       success = simulate_request(action, rps)
       저장: p99_latency, success_rate

   - 시나리오 처리:
     * normal: 원본 RPS 사용
     * spike: Day 97-110 중에 5배 급증 (10초 동안) 주입
      * gradual: 30s baseline → 90s 선형 증가 → 30s peak 유지
      * periodic: 저부하/고부하 10분 주기 반복, transition 플래그 포함

   - 반환: {
       'p99_latency': float,
       'success_rate': float,
       'stability_score': float,
       'adaptation_time': float (spike만),
       'predictive_mae': float (gradual/LSTM),
       'tracking_lag_seconds': float (gradual/LinUCB),
       'detailed_results': pd.DataFrame (scenario 메타데이터 포함)
     }

2. extract_context(rps, error_rate, cpu)
   - 입력: 현재 시스템 상태
   - 반환: 정규화된 context (0-1 범위)

3. simulate_request(throttle_limit, current_rps)
   - 로직: if current_rps <= throttle_limit → True
   - 아니면 확률적으로 일부 허용
```

### 2.5 평가 및 통계

**명세 참조**: @docs/API_DESIGN.md 섹션 2.5

**Codex 지시**:
```
@docs/API_DESIGN.md를 참고하여 src/evaluation.py를 구현해줘.

함수:
1. compute_metrics(detailed_results, scenario)
   - P99: percentile(latencies, 99)
   - Success rate: sum(success) / total
   - Stability: sum(p99 < 150ms) / total
   - Adaptation time (spike만): spike 후 복구 시간

2. run_statistical_tests(model1_results, model2_results)
   - Paired t-test (10회 반복 결과)
   - Effect size (Cohen's d)
   - 신뢰도 구간 (95%)
   - 반환: p_value, significant, effect_size

3. generate_report(all_results)
   - 마크다운 형식
   - 표: 시나리오별 성능
   - 통계: p-value, effect size
   - 결론: 어떤 모델이 더 나은가
```

---

## 3. 병렬 실행 스크립트

**명세 참조**: @docs/IMPLEMENTATION_GUIDE.md 섹션 6

**Codex 지시**:
```
@docs/IMPLEMENTATION_GUIDE.md의 Phase 6을 참고하여
experiments/run_all_scenarios.py를 구현해줘.

요구사항:
1. 120개 실험을 병렬 실행 (샘플 설정은 축약 가능)
   - 4 시나리오 × 3 모델 × 10 seeds
   - multiprocessing.Pool 사용 (8 processes)

2. 각 실험마다:
   - 모델 로드 (또는 학습)
   - 시뮬레이션 실행
   - 결과 저장 (results/{모델}_{시나리오}_{seed}.json)

3. 진행 상황 표시
   - 진행률 바 (tqdm)
   - 완료된 실험 수

4. 통계 분석 자동 실행
   - 모든 실험 완료 후 자동으로 통계 계산
   - 최종 리포트 생성 (results/statistical_summary.txt)
```

---

## 4. Codex 프롬프트 템플릿

### 4.1 기본 템플릿

```markdown
## 구현 요청

파일: `src/[module_name].py`

명세 문서: @docs/[DOCUMENT_NAME].md (섹션 [NUMBER])

요구사항:
1. [함수/클래스 이름]
   - 입력: [타입] [설명]
   - 출력: [타입] [설명]
   - 동작: [구체적 동작 설명]
   - 조건: [추가 조건]

2. [함수/클래스 이름]
   - ...

테스트:
- tests/test_[module_name].py와 모든 테스트가 통과해야 함

추가 사항:
- [특수 요구사항이 있으면]
```

### 4.2 수정 요청 템플릿

```markdown
## 수정 요청

파일: `src/[module_name].py`

함수: `[function_name]`

현재 문제:
[현재 동작의 문제점]

요구 사항:
@docs/[DOCUMENT].md의 [섹션]을 보면...
[명확한 요구사항]

변경 사항:
- [변경할 부분 1]
- [변경할 부분 2]

테스트 확인:
- [관련 테스트] 통과 확인
```

---

## 5. Codex 사용 시 효율 팁

### 5.1 컨텍스트 절감

❌ **비효율적**:
```
"LSTM 모델을 구현하는데...
 - 이건 이래야 하고
 - 저건 저래야 하고
 - 디버깅도 이렇게 해야 하고
 - ..."
```

✅ **효율적**:
```
"@docs/API_DESIGN.md 섹션 2.2를 보고
 LSTMPredictor 클래스를 구현해줘"
```

### 5.2 명확한 성공 기준

❌ **모호함**:
```
"잘 작동하는 코드를 만들어줘"
```

✅ **명확함**:
```
"다음 조건을 만족해야 함:
 - loss < 0.01에서 수렴 (10 epoch 내)
 - predict() 출력 shape: (60,)
 - tests/test_lstm_model.py 모든 테스트 통과"
```

### 5.3 반복적 개선

```
1차: 기본 구현
2차: 성능 최적화 (GPU 활용 등)
3차: 에러 처리 추가
4차: 로깅 및 모니터링 추가
5차: 문서화
```

각 단계별로 Codex에 명확한 지시

---

## 6. 문제 해결 (Troubleshooting)

### 6.1 Codex가 잘못된 코드 생성

**해결책**:
1. @docs/[DOCUMENT].md에서 해당 섹션 다시 읽기
2. 더 구체적인 지시 제공
3. 예시 코드 포함

**예시**:
```
"@docs/API_DESIGN.md 섹션 2.4를 다시 보니
 extract_context() 함수가 정규화를 해야 하는데
 지금 코드에서 정규화가 없어.

 다음과 같이 수정해줘:
 ```python
 context = np.array([...])
 context = (context - min_val) / (max_val - min_val)
 return context
 ```
"
```

### 6.2 명세와 코드 불일치

**해결책**:
1. @docs/SPEC.md에서 명세 재확인
2. 문제 부분 명확히 지적
3. 원하는 동작 설명

### 6.3 테스트 실패

**해결책**:
1. @docs/TESTING_STRATEGY.md의 해당 테스트 참조
2. 실패한 테스트 케이스 설명
3. 수정 방법 제시

```
"tests/test_lstm_model.py의 test_predict_shape()이 실패함.

요구사항:
- predict()의 출력 shape는 (60,)이어야 함
- 현재는 (1, 60)을 반환 중

수정:
- reshape 또는 squeeze 추가"
```

---

## 7. SDD 워크플로우 체크리스트

| 단계 | 작업 | Codex | 체크 |
|------|------|-----------|------|
| 1 | @docs/SPEC.md 작성 | 없음 | [x] |
| 2 | @docs/API_DESIGN.md 작성 | 없음 | [x] |
| 3 | @docs/IMPLEMENTATION_GUIDE.md 작성 | 없음 | [x] |
| 4 | data_pipeline.py 구현 | ✅ | [x] |
| 5 | lstm_model.py 구현 | ✅ | [x] |
| 6 | linucb_agent.py 구현 | ✅ | [x] |
| 7 | simulator.py 구현 | ✅ | [x] |
| 8 | evaluation.py 구현 | ✅ | [x] |
| 9 | 단위 테스트 | ✅ | [x] |
| 10 | 통합 테스트 | 부분 | [ ] |
| 11 | 병렬 실행 스크립트 | ✅ | [x] |
| 12 | 시각화 및 리포팅 | ✅ | [x] |
