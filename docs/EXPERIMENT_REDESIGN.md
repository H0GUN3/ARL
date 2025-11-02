# 실험 재설계 제안서

## 개요
- 목적: 실측 BurstGPT 트래픽을 활용해 Predictive(LSTM)과 Reactive(LinUCB) Rate Limiting 전략을 공정하게 비교하고, 트래픽 패턴별 최적 전략을 제시한다.
- 연구 질문:
  - **RQ1**: 트래픽 패턴(Periodic, Burst, Drift, Failure)에 따라 어떤 전략이 우수한가?
  - **RQ2**: 실무자는 어떤 기준으로 전략을 선택해야 하는가?
- 문제 배경:
  - Static 임계값은 자원 낭비·급증 시 장애를 초래한다.
  - 기존 LSTM/LinUCB 연구는 서로 다른 synthetic workload를 사용해 비교가 불가능했다.
- 실험 방향: 동일 데이터·환경·메트릭에서 LSTM vs LinUCB vs Static을 비교하고, 트래픽 특성에 따른 성능 차이를 정량화한다.
- 참고 근거: BurstGPT 데이터 특성분석(Wang et al., 2024), 대규모 데이터 품질 관리(TFDV)[^tfdv], Google SRE 신뢰성 테스트 권고안[^sre-testing], LinUCB 원저자들의 설계 원칙[^linucb].

## 문제별 개선 전략

### 1. 데이터 준비·분할
- **정교한 데이터 검증 파이프라인**: TFDV 기반으로 통계량/스키마를 생성하고 분포 절단값, 희소도, 범주 누락, 스큐/드리프트 지표까지 자동 감시한다. 전체 파이프라인 초기에 `generate_statistics -> infer_schema -> validate_statistics`를 추가하고, drift 탐지 설정(`float_domain`, `skew_comparator`)으로 시나리오 전후 분포를 비교한다.[^tfdv]
- **산출물 디렉터리 표준화**: TFDV 통계/스키마/이상치 리포트는 `artifacts/tfdv/` (예: `stats.pbtxt`, `schema.pbtxt`, `anomalies.pbtxt`)에 저장해 코드 산출물과 구분하고, `artifacts/README.md`로 파일 용도를 문서화한다.
- **시나리오 실측 기반 추출**: `src/scenario_extraction.py`를 통해 BurstGPT 원본 로그에서 Conversation-Periodic, API-Burst, Gradual-Drift, Failure-Spike 패턴을 직접 추출하고 `scripts/prepare_scenarios.py`로 CSV와 메타데이터를 생성한다. (출력: `data/scenarios/*.csv`.)
- **초 단위 외 다중 해상도 집계**: 초 단위 RPS 외에 100ms·10ms 레벨 이동평균, 최대값을 병행 산출하여 burst 감지를 보존한다. 모델 입력에는 멀티-리졸루션 피처를 추가하고, LinUCB warmup에는 finer-window 통계를 제공한다.
- **중복/불균등 간격 처리**: request_id 기반 중복 제거, 지연 도착 로그 재정렬, 샤딩 타임스탬프 보정 로직을 추가한다. 이상값은 분포 기반 IQR/로버스트 z-score로 표기하고, 제거 대신 별도 플래그로 유지한다.
- **공정한 분할**: 70/10/20 유지하되, concept drift 감시를 위해 기간별 슬라이딩 윈도우 분할(예: 주별)을 병행 저장하고 TFDV drift comparator로 차이를 검증한다.

### 2. 모델 구성 및 학습
- **Static baseline 보강**: 단일 P95 대신, 시간대별(예: 15분) 분포와 최근 burst 히스토리를 반영한 적응형 임계치(예: 이동 quantile + 안전 여유)를 정의하여 현실성을 확보한다. SRE가 권장하는 점진적 롤아웃·가드레일 개념을 baseline에도 적용해 급격한 설정 변경을 방지한다.[^sre-testing]
- **LinUCB 튜닝 근거화**: Li et al.가 제안한 LinUCB-Disjoint 분석에 맞춰 `alpha = O(√(log t))`, 액션 공간 크기, 컨텍스트 정규화 범위를 데이터 분산에 따라 조정한다.[^linucb] Warmup 이후에도 온라인 업데이트를 유지하되, 실제 운영과 동일하게 탐험률 감소 스케줄(예: α_t = α₀ / √t)과 안전 가드레일(성공률 하한선 미달 시 fallback)을 도입한다.
- **LSTM 학습 전략 재구성**:
  - 시나리오별·기간별 균형 샘플링 함수를 도입해 RandomSequenceDataset이 장기 추세/주기/급변 구간을 고르게 포함하도록 한다.
  - 윈도우와 horizon을 실험적으로 탐색하고, hidden unit·dropout·학습률은 검증 세트 기반 베이지안/랜덤 서치로 최적화한다.
  - Early stopping은 상대 개선율(Δ<1%)과 최소 epoch 기준(예: ≥15) 두 조건을 사용하고, 모델별 학습 곡선/스케일러를 버전 관리한다.

#### 모델 재튜닝 실무 계획
- **LinUCB**
  - 탐험계수 스케줄 후보: α₀ ∈ {0.25, 0.5}, 감쇠 함수 α_t = α₀ / √(1 + t / τ) (τ=10⁴ 등) 시뮬레이션
  - 액션 공간: 300–5,000 RPS (50·100 간격) 비교, 안전 가드레일(성공률 <0.85 시 Static fallback) 구현
  - Warmup 정책: 다중 해상도 피처(1s/100ms) + 최근 5분 이동 평균 추가, warmup 길이 5–15% 데이터 비교
- **LSTM**
  - 균형 샘플링: 시나리오 라벨/기간 기반 StratifiedRandomSequenceDataset 구현(Gradual/Spike/Periodic 구간별 동일 샘플 수)
  - 하이퍼파라미터 탐색: window ∈ {60, 90}, horizon ∈ {30, 60}, hidden_units ∈ {(64,32), (96,48)}, dropout ∈ {0.2, 0.3}, learning_rate ∈ {1e-3, 5e-4}
  - EarlyStopping: patience 10, min_delta 0.001, 최소 epoch 20, best 모델과 스케일러 저장 + MLflow 추적

### 3. 시나리오 설계
- **랜덤화·조합 다양화**: Normal/Spike/Gradual/Periodic 각 시나리오에 대해 파라미터 분포(스파이크 시점, 지속시간, 증폭비, 노이즈 스펙트럼 등)를 설정하고, 실험마다 seed별로 표본을 재생성한다. 이를 통해 특정 모델에 유리한 고정 패턴을 제거한다.
- **복합 스트레스 테스트**: SRE 책에서 권장하는 failure injection·watchdog 시나리오처럼, 다중 burst, 단계적 컨셉트 드리프트, 노이즈·오류율 상승을 결합한 “Stress Mix” 시나리오를 신설한다.[^sre-testing]
- **현실 흐름 보강**: 실제 운영에서 관찰 가능한 배치 작업, API priority, 백오프 정책을 반영해 패턴/메타데이터(예: API 클래스, region) 컬럼을 추가하고, 시뮬레이션이 정책 차이를 구분할 수 있게 한다.

### 4. 시뮬레이션 & 평가
- **큐잉/셔틀링 모델링**: 단순 비율 난수 대신 M/M/1·토큰 버킷 기반 큐 시뮬레이터를 도입해 슬로우다운·지연·드롭을 현업 정책과 일치시키고, LinUCB 업데이트는 실제 관측 지연에 맞춰 처리한다.
- **가드레일 메트릭 다중화**: 성공률만이 아니라 p99, p95, backlog length, throttle oscillation, 정책 변동 비용 등을 동시에 기록하고, 가드레일 조건(예: 안정도 <0.8 시 실패)을 정의한다.[^sre-testing]
- **통계적 검증 강화**: 각 시나리오당 seed≥30으로 확장하고, 효과크기(Cohen’s d), 신뢰구간, 검정력(>0.8) 계산을 추가한다. 여러 메트릭에 대해 Holm-Bonferroni 등 다중 가설 보정을 적용하고, 비정규 분포 시 비모수 검정을 병행한다.
- **결과 재현성**: 시나리오 생성 seed·모델 체크포인트·파라미터 로그를 MLflow/Weights & Biases 등 추적 도구에 저장하고, TFDV drift 리포트와 연결해 결과 해석 시 분포 변화 여부를 명시한다.

### 5. 거버넌스·문서화
- **실험 템플릿**: 각 실험 실행 전후에 데이터 프로파일링 보고서, 하이퍼파라미터 근거, 가드레일 정의, 통계 요약을 포함한 체크리스트를 작성한다.
- **검토 프로세스**: 모델별 champion/challenger 평가를 명문화하고, baseline 대비 우월성 판정을 가드레일 + 효과크기 + 비용 기준으로 통합한다.
- **문서 업데이트**: `docs/PHASE.md`에 “Phase 6. 실험 재설계 실행”을 추가하고, README에는 Stress Mix 시나리오/다중 해상도 파이프라인을 반영한다.

## 실행 로드맵
1. **Phase 1 – Baseline 재확인 (주 1)**  
   - 기존 설정으로 4시나리오 × 3모델 × 5seed 실험 재현, 차이가 미미함을 문서화.  
   - BurstGPT 실측 시나리오(`data/scenarios/`)가 기존 synthetic 시나리오와 특성이 어떻게 다른지 검증.
2. **Phase 2 – 최소 개선 적용 (주 2-3)**  
   - LinUCB: 컨텍스트 3→6개(급증 감지, 시간대 포함), 탐험률 감쇠 `α_t=α₀/√(1+t/τ)`, 액션 계층화.  
   - LSTM: Stratified 샘플링, 주요 하이퍼파라미터 탐색(window, horizon, hidden_units, dropout).  
   - 시나리오: Spike/Gradual/Periodic 강도를 BurstGPT 통계에 맞게 확대하여 전략별 차이가 드러나도록 조정.
3. **Phase 3 – 실험 실행 & 분석 (주 4)**  
   - 개선된 구성으로 4시나리오 × 3모델 × seed≥5 실험 수행.  
   - 메트릭(성공률, p99, 적응 시간, 안정도, 실패 복구)을 비교하고 paired t-test, Cohen’s d 분석.
4. **Phase 4 – 논문화 (주 5-6)**  
   - 트래픽 패턴별 전략 우위, 의사결정 가이드라인 정리.  
   - `docs/REPORT_DRAFT.md`·`README.md` 업데이트, `docs/EXPERIMENT_VERSIONS.md`에 v2 결과 기록.

## 참고 자료
- [^tfdv]: TensorFlow Data Validation Guide – “TFDV can compute descriptive statistics, infer schema, and detect anomalies/drift for large-scale datasets” (https://www.tensorflow.org/tfx/data_validation/get_started)
- [^sre-testing]: Google Site Reliability Engineering Book, “Testing for Reliability” – 권장되는 점진적 배포, 실패 주입, 가드레일 기반 신뢰성 확보 전략 (https://landing.google.com/sre/sre-book/chapters/testing-reliability/)
- [^linucb]: Li, Chu, Langford, Schapire. *A Contextual-Bandit Approach to Personalized News Article Recommendation* – LinUCB 탐험계수 설정과 불확실도 범위 계산 방식을 제시 (https://arxiv.org/abs/1003.0146)
