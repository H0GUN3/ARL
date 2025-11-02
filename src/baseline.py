"""
기준선 Rate Limiter 구현.

단순히 학습 데이터의 RPS P95 값을 throttle limit으로 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StaticRateLimiter:
    """고정 임계값 기반 Rate Limiter."""

    threshold: float

    @classmethod
    def from_data(cls, train_data: pd.DataFrame) -> "StaticRateLimiter":
        if "rps" not in train_data.columns:
            raise ValueError("train_data에 'rps' 컬럼이 필요합니다.")
        threshold = float(np.percentile(train_data["rps"], 95))
        return cls(threshold=threshold)

    def select_action(self, context) -> int:
        return int(max(self.threshold, 1.0))
