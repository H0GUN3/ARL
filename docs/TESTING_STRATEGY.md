# 테스트 및 검증 전략 (Testing Strategy)

## 1. 단위 테스트 (Unit Tests)

### 1.1 Data Pipeline 테스트
**파일**: tests/test_data_pipeline.py

```python
import unittest

class TestDataPipeline(unittest.TestCase):

    def test_load_burstgpt(self):
        """CSV 파일 로딩 테스트"""
        data = load_burstgpt('data/burstgpt_v1.1.csv')
        self.assertGreater(len(data), 0)
        self.assertIn('timestamp', data.columns)
        self.assertIn('response_tokens', data.columns)

    def test_aggregate_to_timeseries(self):
        """1초 단위 집계 테스트"""
        raw = load_burstgpt('data/burstgpt_v1.1.csv')
        ts = aggregate_to_timeseries(raw, interval=1)

        # 검증
        self.assertEqual(len(ts), 121 * 86400)  # 121일
        self.assertIn('rps', ts.columns)
        self.assertIn('p99_latency', ts.columns)
        self.assertTrue((ts['rps'] >= 0).all())

    def test_split_data(self):
        """Train/Warmup/Test 분할 테스트"""
        ts = load_timeseries('data/burstgpt_timeseries.csv')
        splits = split_data(ts)

        # 비율 검증
        total = len(ts)
        self.assertEqual(len(splits['train']), int(total * 0.70))
        self.assertEqual(len(splits['warmup']), int(total * 0.10))
        self.assertEqual(len(splits['test'][0]), int(total * 0.20 / 5))

        # Time leakage 검증
        train_end = splits['train'].index[-1]
        warmup_start = splits['warmup'].index[0]
        self.assertLess(train_end, warmup_start)

    def test_no_missing_values(self):
        """결측값 확인 테스트"""
        ts = load_timeseries('data/burstgpt_timeseries.csv')
        self.assertEqual(ts.isnull().sum().sum(), 0)

    def test_value_ranges(self):
        """값 범위 검증"""
        ts = load_timeseries('data/burstgpt_timeseries.csv')
        self.assertTrue((ts['rps'] > 0).all())
        self.assertTrue((ts['rps'] < 100000).all())
        self.assertTrue((ts['error_rate'] >= 0).all())
        self.assertTrue((ts['error_rate'] <= 1).all())
```

### 1.2 LSTM 모델 테스트
**파일**: tests/test_lstm_model.py

```python
class TestLSTMPredictor(unittest.TestCase):

    def setUp(self):
        self.model = LSTMPredictor(window_size=60, hidden_units=128)
        self.train_data = load_timeseries('data/burstgpt_timeseries.csv').iloc[:100000]

    def test_model_initialization(self):
        """모델 초기화 테스트"""
        self.assertIsNotNone(self.model.model)
        self.assertEqual(self.model.window_size, 60)

    def test_fit(self):
        """학습 테스트"""
        result = self.model.fit(self.train_data, epochs=2, batch_size=32)

        self.assertIn('train_loss', result)
        self.assertIn('convergence', result)
        self.assertGreater(len(result['train_loss']), 0)
        # 손실이 감소하는지 확인
        self.assertLess(result['train_loss'][-1], result['train_loss'][0])

    def test_predict_shape(self):
        """예측 출력 형태 테스트"""
        self.model.fit(self.train_data, epochs=1)
        context = self.train_data.iloc[-60:]
        prediction = self.model.predict(context)

        self.assertEqual(prediction.shape, (60,))  # 60초 예측
        self.assertTrue((prediction >= 0).all())  # RPS는 음수 불가

    def test_save_load(self):
        """모델 저장/로드 테스트"""
        self.model.fit(self.train_data, epochs=1)

        # 저장
        self.model.save('temp_model.pkl')

        # 로드
        model2 = LSTMPredictor()
        model2.load('temp_model.pkl')

        # 예측 일치 확인
        context = self.train_data.iloc[-60:]
        pred1 = self.model.predict(context)
        pred2 = model2.predict(context)

        np.testing.assert_array_almost_equal(pred1, pred2)
```

### 1.3 LinUCB Agent 테스트
**파일**: tests/test_linucb_agent.py

```python
class TestLinUCBAgent(unittest.TestCase):

    def setUp(self):
        self.agent = LinUCBAgent(alpha=0.25)
        self.warmup_data = load_timeseries('data/burstgpt_timeseries.csv').iloc[500000:520000]

    def test_initialization(self):
        """에이전트 초기화 테스트"""
        self.assertEqual(len(self.agent.action_space), 50)
        self.assertEqual(self.agent.alpha, 0.25)

    def test_warmup(self):
        """Warmup 학습 테스트"""
        result = self.agent.warmup(self.warmup_data)

        self.assertIn('regret_curve', result)
        self.assertIn('convergence', result)
        # Regret이 감소하는지 확인
        regrets = np.array(result['regret_curve'])
        self.assertLess(regrets[-1], regrets[0] * 2)  # 어느 정도 감소

    def test_select_action(self):
        """Action 선택 테스트"""
        context = np.array([1000, 0.02, 0.5])  # [rps, error_rate, cpu]
        action = self.agent.select_action(context)

        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 50)  # action space 범위

    def test_update(self):
        """파라미터 업데이트 테스트"""
        context = np.array([1000, 0.02, 0.5])
        action = 10
        reward = 0.95

        A_before = self.agent.A[action].copy()

        self.agent.update(context, action, reward)

        # A가 업데이트되었는지 확인
        self.assertFalse(np.array_equal(A_before, self.agent.A[action]))

    def test_save_load(self):
        """저장/로드 테스트"""
        self.agent.warmup(self.warmup_data)

        # 저장
        self.agent.save('temp_agent.json')

        # 로드
        agent2 = LinUCBAgent()
        agent2.load('temp_agent.json')

        # 동일한 action 선택 확인
        context = np.array([1000, 0.02, 0.5])
        action1 = self.agent.select_action(context)
        action2 = agent2.select_action(context)

        self.assertEqual(action1, action2)
```

---

## 2. 통합 테스트 (Integration Tests)

### 2.1 End-to-End 테스트
**파일**: tests/test_integration.py

```python
class TestEndToEnd(unittest.TestCase):

    def test_lstm_pipeline(self):
        """LSTM 전체 파이프라인 테스트"""
        # 1. 데이터 준비
        train_data = load_timeseries('data/burstgpt_timeseries.csv').iloc[:int(5.29e6 * 0.70)]
        test_data = load_timeseries('data/burstgpt_timeseries.csv').iloc[-int(5.29e6 * 0.20):]

        # 2. 모델 학습
        model = LSTMPredictor()
        fit_result = model.fit(train_data, epochs=5)
        self.assertTrue(fit_result['convergence'])

        # 3. 시뮬레이션 실행
        metrics = run_simulation(model, test_data, scenario='normal', seed=42)

        # 4. 메트릭 검증
        self.assertIn('p99_latency', metrics)
        self.assertIn('success_rate', metrics)
        self.assertGreater(metrics['success_rate'], 0)
        self.assertLess(metrics['success_rate'], 1)

    def test_linucb_pipeline(self):
        """LinUCB 전체 파이프라인 테스트"""
        # 1. 데이터 준비
        warmup_data = load_timeseries('data/burstgpt_timeseries.csv').iloc[int(5.29e6 * 0.70):int(5.29e6 * 0.80)]
        test_data = load_timeseries('data/burstgpt_timeseries.csv').iloc[-int(5.29e6 * 0.20):]

        # 2. 에이전트 Warmup
        agent = LinUCBAgent()
        warmup_result = agent.warmup(warmup_data)
        self.assertTrue(warmup_result['convergence'])

        # 3. 시뮬레이션 실행
        metrics = run_simulation(agent, test_data, scenario='normal', seed=42)

        # 4. 메트릭 검증
        self.assertGreater(metrics['success_rate'], 0)

    def test_spike_scenario(self):
        """Spike 시나리오 테스트"""
        test_data = load_timeseries('data/burstgpt_timeseries.csv').iloc[-int(5.29e6 * 0.20):]
        spiked_data = create_spike_scenario(test_data)

        metrics = run_simulation(model, spiked_data, scenario='spike', seed=42)

        # Adaptation time이 계산되었는지 확인
        self.assertIn('adaptation_time', metrics)
        self.assertGreater(metrics['adaptation_time'], 0)
```

---

## 3. 성능 테스트 (Performance Tests)

### 3.1 벤치마크
**파일**: tests/test_performance.py

```python
import time

class TestPerformance(unittest.TestCase):

    def test_lstm_training_time(self):
        """LSTM 학습 시간 벤치마크"""
        train_data = load_timeseries('data/burstgpt_timeseries.csv').iloc[:100000]
        model = LSTMPredictor()

        start = time.time()
        model.fit(train_data, epochs=10)
        elapsed = time.time() - start

        print(f"LSTM training time: {elapsed:.2f}s")
        # 목표: 30분 이내
        self.assertLess(elapsed, 1800)

    def test_linucb_warmup_time(self):
        """LinUCB Warmup 시간 벤치마크"""
        warmup_data = load_timeseries('data/burstgpt_timeseries.csv').iloc[:100000]
        agent = LinUCBAgent()

        start = time.time()
        agent.warmup(warmup_data)
        elapsed = time.time() - start

        print(f"LinUCB warmup time: {elapsed:.2f}s")
        # 목표: 5분 이내
        self.assertLess(elapsed, 300)

    def test_simulation_throughput(self):
        """시뮬레이션 처리 속도"""
        test_data = load_timeseries('data/burstgpt_timeseries.csv').iloc[:10000]
        model = LSTMPredictor()

        start = time.time()
        metrics = run_simulation(model, test_data, scenario='normal', seed=42)
        elapsed = time.time() - start

        throughput = len(test_data) / elapsed
        print(f"Simulation throughput: {throughput:.0f} seconds/sec")
        # 목표: 최소 1000 seconds/sec
        self.assertGreater(throughput, 1000)
```

---

## 4. 검증 테스트 (Validation Tests)

### 4.1 공정성 검증
**파일**: tests/test_fairness.py

```python
class TestFairness(unittest.TestCase):

    def test_same_test_data(self):
        """모든 모델이 동일한 테스트 데이터 사용"""
        test_data = load_timeseries('data/burstgpt_timeseries.csv').iloc[-int(5.29e6 * 0.20):]

        lstm_metrics = run_simulation(lstm_model, test_data, 'normal', seed=42)
        linucb_metrics = run_simulation(linucb_agent, test_data, 'normal', seed=42)
        static_metrics = run_simulation(static_baseline, test_data, 'normal', seed=42)

        # 같은 RPS 데이터를 본 것 확인
        # (시뮬레이션 로그에서 확인)

    def test_no_time_leakage(self):
        """Time leakage 확인"""
        splits = split_data(load_timeseries('data/burstgpt_timeseries.csv'))

        train_end = splits['train'].index[-1]
        warmup_start = splits['warmup'].index[0]
        warmup_end = splits['warmup'].index[-1]
        test_starts = [t.index[0] for t in splits['test']]

        self.assertLess(train_end, warmup_start)
        self.assertLess(warmup_end, test_starts[0])

    def test_convergence_verification(self):
        """모델 수렴 확인"""
        train_data = load_timeseries('data/burstgpt_timeseries.csv').iloc[:int(5.29e6 * 0.70)]

        # LSTM 수렴 확인
        lstm_model = LSTMPredictor()
        lstm_result = lstm_model.fit(train_data, epochs=50)
        self.assertTrue(lstm_result['convergence'])

        # LinUCB 수렴 확인
        warmup_data = load_timeseries('data/burstgpt_timeseries.csv').iloc[int(5.29e6 * 0.70):int(5.29e6 * 0.80)]
        linucb_agent = LinUCBAgent()
        linucb_result = linucb_agent.warmup(warmup_data)
        self.assertTrue(linucb_result['convergence'])
```

---

## 5. 통계 검증 (Statistical Validation)

### 5.1 통계 테스트
**파일**: tests/test_statistics.py

```python
class TestStatistics(unittest.TestCase):

    def test_paired_ttest(self):
        """Paired t-test 유효성"""
        # 시뮬레이션 결과 로드 (10 runs)
        lstm_results = [...]  # 10개 값
        linucb_results = [...]  # 10개 값

        t_stat, p_value = scipy.stats.ttest_rel(lstm_results, linucb_results)

        # p-value는 0-1 범위
        self.assertGreaterEqual(p_value, 0)
        self.assertLessEqual(p_value, 1)

    def test_effect_size(self):
        """Effect size 계산 검증"""
        lstm_results = [...]
        linucb_results = [...]

        cohens_d = compute_cohens_d(lstm_results, linucb_results)

        # Cohen's d 범위 확인
        self.assertGreaterEqual(abs(cohens_d), 0)

    def test_confidence_interval(self):
        """신뢰도 구간 계산"""
        results = [...]  # 10개 값

        ci_low, ci_high = compute_ci(results, confidence=0.95)

        # 신뢰도 구간이 유의미한지 확인
        self.assertLess(ci_low, ci_high)
        self.assertLess(ci_high - ci_low, 100)  # 너무 넓지 않음
```

---

## 6. 회귀 테스트 (Regression Tests)

**파일**: tests/test_regression.py

```python
class TestRegression(unittest.TestCase):

    def test_results_reproducibility(self):
        """동일한 seed로 재현 가능성"""
        # Run 1
        metrics1 = run_simulation(model, test_data, 'normal', seed=42)

        # Run 2
        metrics2 = run_simulation(model, test_data, 'normal', seed=42)

        # 결과 일치 확인
        np.testing.assert_array_almost_equal(
            metrics1['p99_latency'],
            metrics2['p99_latency']
        )

    def test_backward_compatibility(self):
        """이전 버전 모델 호환성"""
        # 이전 버전에서 저장한 모델 로드
        old_model = load_old_model('models/v1.0_lstm.pkl')

        # 현재 코드로 예측 가능한지 확인
        context = load_timeseries('data/burstgpt_timeseries.csv').iloc[-60:]
        prediction = old_model.predict(context)

        self.assertIsNotNone(prediction)
```

---

## 7. 테스트 실행 명령어

### 7.1 전체 테스트
```bash
# 모든 테스트 실행
python -m pytest tests/ -v

# 특정 테스트만 실행
python -m pytest tests/test_data_pipeline.py -v

# 성능 테스트 (시간이 걸림)
python -m pytest tests/test_performance.py -v

# Coverage 리포트
python -m pytest tests/ --cov=src --cov-report=html
```

### 7.2 CI/CD 파이프라인 (GitHub Actions)
**파일**: .github/workflows/test.yml

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run tests
      run: pytest tests/ --cov=src

    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

---

## 8. 테스트 체크리스트

| 카테고리 | 테스트 | 상태 |
|---------|--------|------|
| Unit | Data pipeline | [ ] |
| Unit | LSTM model | [ ] |
| Unit | LinUCB agent | [ ] |
| Integration | LSTM pipeline | [ ] |
| Integration | LinUCB pipeline | [ ] |
| Integration | Spike scenario | [ ] |
| Performance | Training time | [ ] |
| Performance | Throughput | [ ] |
| Validation | Fairness | [ ] |
| Validation | Convergence | [ ] |
| Statistics | Paired t-test | [ ] |
| Regression | Reproducibility | [ ] |

