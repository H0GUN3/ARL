import numpy as np
import pandas as pd

from src.linucb_agent import LinUCBAgent


def synthetic_warmup(rows: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        {
            "rps": rng.uniform(200, 4000, size=rows),
            "error_rate": rng.uniform(0, 0.2, size=rows),
            "cpu_percent": rng.uniform(20, 90, size=rows),
        }
    )
    return data


def test_select_and_update():
    agent = LinUCBAgent(context_keys=["rps", "error_rate", "cpu_percent"], max_rps=4000.0)
    context = np.array([0.5, 0.1, 0.4], dtype=float)
    idx = agent.select_action(context)
    assert 0 <= idx < len(agent.action_space)
    agent.update(context, idx, reward=0.8)
    assert agent.action_counts[idx] == 1


def test_warmup_and_save_load(tmp_path):
    data = synthetic_warmup(150)
    agent = LinUCBAgent(context_keys=["rps", "error_rate", "cpu_percent"], max_rps=4000.0, alpha_decay=True, decay_tau=1000.0)
    summary = agent.warmup(data)

    assert len(summary.regret_curve) == len(data)
    assert isinstance(summary.convergence, bool)
    assert sum(summary.action_counts.values()) == len(data)

    save_path = tmp_path / "linucb.json"
    agent.save(save_path)

    restored = LinUCBAgent(context_keys=["rps", "error_rate", "cpu_percent"], max_rps=4000.0)
    restored.load(save_path)
    idx = restored.select_action(np.array([0.1, 0.0, 0.5], dtype=float))
    assert 0 <= idx < len(restored.action_space)
