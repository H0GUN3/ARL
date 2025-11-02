import numpy as np
import pandas as pd
from src.baseline import StaticRateLimiter
from src.linucb_agent import LinUCBAgent
from src.lstm_model import LSTMPredictor
from src.scenario_generator import (
    create_gradual_scenario,
    create_normal_scenario,
)
from src.simulator import run_simulation, simulate_request


def test_simulate_request_probability():
    rng = np.random.default_rng(0)
    throttle = 100
    current = 200
    outcomes = [simulate_request(throttle, current, rng) for _ in range(1000)]
    mean = np.mean(outcomes)
    assert 0 < mean < 1


def test_run_simulation_with_static_model():
    data = create_normal_scenario(mean_rps=700, duration=240)
    static = StaticRateLimiter.from_data(data)
    result = run_simulation(static, data, scenario="normal", seed=0)
    assert 0 <= result.success_rate <= 1
    assert isinstance(result.detailed_results, pd.DataFrame)
    assert not result.detailed_results.empty
    assert result.predictive_mae is None
    assert result.tracking_lag_seconds is None


def test_run_simulation_with_linucb():
    data = create_normal_scenario(mean_rps=600, duration=200)
    agent = LinUCBAgent(context_keys=["rps", "error_rate", "cpu_percent"], max_rps=6000.0)
    agent.warmup(data.head(120))
    result = run_simulation(agent, data.tail(80), scenario="normal", seed=1)
    assert 0 <= result.success_rate <= 1
    assert result.detailed_results.shape[0] == 80
    assert result.tracking_lag_seconds is not None


def test_run_simulation_with_lstm_predictions():
    train_data = create_normal_scenario(mean_rps=900, duration=400)
    lstm = LSTMPredictor(window_size=20, horizon=5, hidden_units_1=32, hidden_units_2=16, dropout=0.1, learning_rate=1e-3)
    lstm.fit(train_data, epochs=1, batch_size=32, samples_per_epoch=128)
    scenario = create_gradual_scenario()
    result = run_simulation(lstm, scenario, scenario="gradual", seed=2)
    assert result.predictive_mae is not None
