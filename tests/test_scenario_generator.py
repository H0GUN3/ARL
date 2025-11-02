import numpy as np

from src.scenario_generator import (
    create_gradual_scenario,
    create_normal_scenario,
    create_periodic_scenario,
    create_spike_scenario,
)


def test_normal_scenario_properties():
    df = create_normal_scenario(mean_rps=900, duration=180, seed=1)
    assert (df["scenario"] == "normal").all()
    assert (df["scenario_label"] == "normal").all()
    cv = df["rps"].std() / df["rps"].mean()
    assert cv < 0.15
    assert df["cpu_percent"].between(0, 100).all()


def test_spike_scenario_multiplier():
    df = create_spike_scenario(base_rps=400, spike_multiplier=8, seed=1)
    baseline = df.iloc[:60]["rps"].mean()
    peak = df["rps"].max()
    assert peak / baseline >= 7.0  # 허용된 하한
    assert df["is_spike"].all()
    assert (df["scenario_label"] == "spike").all()

    df_intense = create_spike_scenario(base_rps=400, spike_multiplier=10, duration_scale=0.5, noise_scale=1.2, seed=1)
    peak2 = df_intense["rps"].max()
    baseline2 = df_intense.iloc[: int(60 * 0.5)]["rps"].mean()
    assert peak2 / baseline2 >= 9.0


def test_gradual_scenario_linearity():
    df = create_gradual_scenario()
    ramp_duration = len(df[df["phase"] == "ramp"])
    ramp = df[df["phase"] == "ramp"]
    x = range(len(ramp))
    slope, _ = np.polyfit(x, ramp["rps"], 1)
    assert 20 <= slope <= 40
    assert set(df["phase"]) == {"baseline", "ramp", "peak"}
    assert (df["scenario_label"] == "gradual").all()

    df_extended = create_gradual_scenario(base_rps=400, target_multiplier=6.0, ramp_duration=60, noise_scale=1.1)
    assert df_extended["rps"].max() > 2000
    assert df_extended["phase"].value_counts().loc["ramp"] == 60


def test_periodic_scenario_autocorr():
    df = create_periodic_scenario()
    period = int(df["period"].iloc[0])
    autocorr = df["rps"].autocorr(lag=period)
    assert autocorr > 0.6
    assert set(df["phase"]) == {"low", "high"}
    assert (df["scenario_label"] == "periodic").all()

    df_ratio = create_periodic_scenario(low_rps=500, high_rps=1500, ratio=3.5, noise_scale=1.2)
    low_mean = df_ratio[df_ratio["phase"] == "low"]["rps"].mean()
    high_mean = df_ratio[df_ratio["phase"] == "high"]["rps"].mean()
    assert 3.0 <= high_mean / max(low_mean, 1) <= 4.5
