"""
LinUCB 기반 Rate Limiting 에이전트 구현.

명세(@docs/API_DESIGN.md)에 따라 컨텍스트(초당 요청수, 에러율, CPU 사용률)를
입력으로 받아 throttle action을 선택하고 파라미터를 온라인으로 업데이트한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def _default_action_space() -> List[int]:
    # Tiered action space to balance exploration speed and control granularity.
    return [
        300,
        400,
        500,
        600,
        800,
        1000,
        1200,
        1500,
        1800,
        2100,
        2500,
        3000,
        3600,
        4200,
        5000,
    ]


@dataclass
class WarmupSummary:
    regret_curve: List[float]
    convergence: bool
    final_params: Dict[str, float]
    action_counts: Dict[int, int]


class LinUCBAgent:
    """LinUCB 컨텍스트 밴딧 에이전트."""

    def __init__(
        self,
        action_space: Iterable[int] | None = None,
        alpha: float = 0.25,
        context_keys: Iterable[str] | None = None,
        alpha_decay: bool = False,
        decay_tau: float = 10_000.0,
        min_alpha: float = 0.05,
        max_rps: float = 5_000.0,
        regularization: float = 1.0,
    ) -> None:
        self.action_space = list(action_space) if action_space is not None else _default_action_space()
        self.context_keys = list(
            context_keys
            if context_keys is not None
            else ["rps", "error_rate", "cpu_percent", "rps_delta_5s", "rps_std_30s", "time_of_day_sin", "time_of_day_cos"]
        )
        self.d = len(self.context_keys)
        self.regularization = regularization
        self.max_rps = max_rps
        self.A = {idx: np.identity(self.d) * regularization for idx, _ in enumerate(self.action_space)}
        self.b = {idx: np.zeros(self.d) for idx, _ in enumerate(self.action_space)}
        self.action_counts = {idx: 0 for idx in range(len(self.action_space))}
        self.alpha0 = alpha
        self.use_alpha_decay = alpha_decay
        self.decay_tau = decay_tau
        self.min_alpha = min_alpha
        self._step = 0

    def _extract_context(self, row: pd.Series) -> np.ndarray:
        values: List[float] = []
        for key in self.context_keys:
            if key == "rps":
                rps = float(row.get("rps", 0.0))
                values.append(np.clip(rps / self.max_rps, 0.0, 1.5))
            elif key == "error_rate":
                values.append(np.clip(float(row.get("error_rate", 0.0)), 0.0, 1.0))
            elif key in {"cpu_percent", "cpu"}:
                cpu = float(row.get(key, row.get("cpu", 0.0))) / 100.0
                values.append(np.clip(cpu, 0.0, 1.0))
            elif key == "rps_delta_5s":
                values.append(np.clip(float(row.get(key, 0.0)), -1.5, 1.5))
            elif key == "rps_std_30s":
                values.append(np.clip(float(row.get(key, 0.0)), 0.0, 1.5))
            elif key == "time_of_day_sin":
                values.append(float(row.get(key, 0.0)))
            elif key == "time_of_day_cos":
                values.append(float(row.get(key, 0.0)))
            else:
                values.append(float(row.get(key, 0.0)))
        return np.asarray(values, dtype=float)

    def _current_alpha(self) -> float:
        if not self.use_alpha_decay:
            return self.alpha0
        scaled = self.alpha0 / np.sqrt(1.0 + self._step / max(self.decay_tau, 1.0))
        return float(max(self.min_alpha, scaled))

    def _compute_reward(self, row: pd.Series, action_idx: int) -> float:
        base_reward = 1.0 - float(row.get("error_rate", 0.0))
        base_reward = np.clip(base_reward, 0.0, 1.0)
        rps = float(row.get("rps", 0.0))
        throttle = self.action_space[action_idx]
        utilization = min(rps / max(throttle, 1.0), 1.5)
        balance = 1.0 - max(utilization - 1.0, 0.0)
        reward = 0.7 * base_reward + 0.3 * balance
        return float(np.clip(reward, 0.0, 1.0))

    def select_action(self, context: np.ndarray) -> int:
        scores = []
        alpha = self._current_alpha()
        for idx in range(len(self.action_space)):
            A_inv = np.linalg.inv(self.A[idx])
            theta = A_inv @ self.b[idx]
            exploit = theta @ context
            explore = alpha * np.sqrt(context @ A_inv @ context)
            scores.append(exploit + explore)
        action_idx = int(np.argmax(scores))
        return action_idx

    def update(self, context: np.ndarray, action_idx: int, reward: float) -> None:
        context = context.reshape(-1, 1)
        self.A[action_idx] += context @ context.T
        self.b[action_idx] += reward * context.flatten()
        self.action_counts[action_idx] += 1
        self._step += 1

    def warmup(self, warmup_data: pd.DataFrame) -> WarmupSummary:
        regret_curve: List[float] = []
        cumulative_regret = 0.0

        for _, row in warmup_data.iterrows():
            context = self._extract_context(row)
            action_idx = self.select_action(context)
            reward = self._compute_reward(row, action_idx)

            self.update(context, action_idx, reward)

            regret = max(0.0, 1.0 - reward)
            cumulative_regret += regret
            regret_curve.append(cumulative_regret)

        convergence = False
        if len(regret_curve) >= 1000:
            recent = regret_curve[-1000:]
            convergence = (recent[0] - recent[-1]) <= 0.0
        else:
            convergence = cumulative_regret / max(len(regret_curve), 1) < 0.05

        final_params = {
            "mean_reward_estimate": float(
                np.mean([np.linalg.norm(self.b[idx]) for idx in range(len(self.action_space))])
            ),
            "total_updates": float(sum(self.action_counts.values())),
        }

        return WarmupSummary(
            regret_curve=regret_curve,
            convergence=convergence,
            final_params=final_params,
            action_counts={self.action_space[idx]: count for idx, count in self.action_counts.items()},
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        state = {
            "action_space": self.action_space,
            "alpha0": self.alpha0,
            "d": self.d,
            "regularization": self.regularization,
            "context_keys": self.context_keys,
            "max_rps": self.max_rps,
            "use_alpha_decay": self.use_alpha_decay,
            "decay_tau": self.decay_tau,
            "min_alpha": self.min_alpha,
            "step": self._step,
            "A": {str(idx): self.A[idx].tolist() for idx in self.A},
            "b": {str(idx): self.b[idx].tolist() for idx in self.b},
        }
        path.write_text(json.dumps(state))

    def load(self, path: str | Path) -> None:
        state = json.loads(Path(path).read_text())
        self.action_space = list(state["action_space"])
        self.alpha0 = float(state.get("alpha0", state.get("alpha", 0.25)))
        self.context_keys = list(state.get("context_keys", self.context_keys))
        self.d = len(self.context_keys)
        self.regularization = float(state.get("regularization", 1.0))
        self.max_rps = float(state.get("max_rps", self.max_rps))
        self.use_alpha_decay = bool(state.get("use_alpha_decay", self.use_alpha_decay))
        self.decay_tau = float(state.get("decay_tau", self.decay_tau))
        self.min_alpha = float(state.get("min_alpha", self.min_alpha))
        self._step = int(state.get("step", 0))
        self.A = {int(k): np.array(v) for k, v in state["A"].items()}
        self.b = {int(k): np.array(v) for k, v in state["b"].items()}
        self.action_counts = {idx: 0 for idx in range(len(self.action_space))}

    def get_action_value(self, action_idx: int) -> int:
        return self.action_space[action_idx]
