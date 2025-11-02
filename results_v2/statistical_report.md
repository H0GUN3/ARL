# 실험 결과 요약

## 시나리오별 성능 요약

### Burst
```
        success_rate  p99_latency  stability_score  adaptation_time  predictive_mae  tracking_lag_seconds  pattern_recognition  proactive_adjustment
model                                                                                                                                               
LSTM           0.630      150.830            0.374            0.000         341.303                   NaN                  NaN                   NaN
LSTM           0.640      150.830            0.412            0.000         341.303                   NaN                  NaN                   NaN
LSTM           0.645      150.830            0.398            0.000         341.303                   NaN                  NaN                   NaN
LinUCB         0.749      150.830            0.370            0.000             NaN                99.000                  NaN                   NaN
LinUCB         0.763      150.830            0.374            0.000             NaN                99.000                  NaN                   NaN
LinUCB         0.763      150.830            0.379            0.000             NaN                99.000                  NaN                   NaN
Static         0.877      150.830            0.422            0.000             NaN                   NaN                  NaN                   NaN
Static         0.867      150.830            0.441            0.000             NaN                   NaN                  NaN                   NaN
Static         0.891      150.830            0.460            0.000             NaN                   NaN                  NaN                   NaN
```

### Drift
```
        success_rate  p99_latency  stability_score  adaptation_time  predictive_mae  tracking_lag_seconds  pattern_recognition  proactive_adjustment
model                                                                                                                                               
LSTM           0.733      171.575            0.589              NaN         816.168                   NaN                  NaN                   NaN
LSTM           0.711      171.575            0.611              NaN         816.168                   NaN                  NaN                   NaN
LSTM           0.689      171.575            0.633              NaN         816.168                   NaN                  NaN                   NaN
LinUCB         0.567      171.575            0.444              NaN             NaN                56.000                  NaN                   NaN
LinUCB         0.611      171.575            0.422              NaN             NaN                56.000                  NaN                   NaN
LinUCB         0.544      171.575            0.478              NaN             NaN                56.000                  NaN                   NaN
Static         0.689      171.575            0.511              NaN             NaN                   NaN                  NaN                   NaN
Static         0.689      171.575            0.500              NaN             NaN                   NaN                  NaN                   NaN
Static         0.678      171.575            0.533              NaN             NaN                   NaN                  NaN                   NaN
```

### Failure
```
        success_rate  p99_latency  stability_score  adaptation_time  predictive_mae  tracking_lag_seconds  pattern_recognition  proactive_adjustment
model                                                                                                                                               
LSTM           1.000      145.176            1.000              NaN         172.548                   NaN                  NaN                   NaN
LSTM           1.000      145.176            1.000              NaN         172.548                   NaN                  NaN                   NaN
LSTM           1.000      145.176            1.000              NaN         172.548                   NaN                  NaN                   NaN
LinUCB         1.000      145.176            1.000              NaN             NaN                 0.000                  NaN                   NaN
LinUCB         1.000      145.176            1.000              NaN             NaN                 0.000                  NaN                   NaN
LinUCB         1.000      145.176            1.000              NaN             NaN                 0.000                  NaN                   NaN
Static         1.000      145.176            1.000              NaN             NaN                   NaN                  NaN                   NaN
Static         1.000      145.176            1.000              NaN             NaN                   NaN                  NaN                   NaN
Static         1.000      145.176            1.000              NaN             NaN                   NaN                  NaN                   NaN
```

### Periodic
```
        success_rate  p99_latency  stability_score  adaptation_time  predictive_mae  tracking_lag_seconds  pattern_recognition  proactive_adjustment
model                                                                                                                                               
LSTM           0.584      151.900            0.295              NaN         330.604                   NaN               -0.870                 0.004
LSTM           0.606      151.900            0.335              NaN         330.604                   NaN               -0.870                 0.004
LSTM           0.600      151.900            0.343              NaN         330.604                   NaN               -0.870                 0.004
LinUCB         0.723      151.900            0.319              NaN             NaN               249.000               -0.870                 0.004
LinUCB         0.767      151.900            0.339              NaN             NaN               249.000               -0.870                 0.004
LinUCB         0.757      151.900            0.345              NaN             NaN               249.000               -0.870                 0.004
Static         0.863      151.900            0.396              NaN             NaN                   NaN               -0.870                 0.004
Static         0.871      151.900            0.388              NaN             NaN                   NaN               -0.870                 0.004
Static         0.865      151.900            0.388              NaN             NaN                   NaN               -0.870                 0.004
```

## 통계 검정 결과
- success_rate (periodic) LSTM vs LinUCB: mean=0.597 vs 0.749, diff=-0.152, t=-22.367, p=0.002, d=-12.914, n=3 → 유의
- success_rate (periodic) LSTM vs Static: mean=0.597 vs 0.867, diff=-0.270, t=-57.571, p=0.000, d=-33.239, n=3 → 유의
- success_rate (periodic) LinUCB vs Static: mean=0.749 vs 0.867, diff=-0.118, t=-10.300, p=0.009, d=-5.946, n=3 → 유의
- p99_latency (periodic) LSTM vs LinUCB: mean=151.900 vs 151.900, diff=0.000, t=nan, p=nan, d=0.000, n=3 → 비유의
- p99_latency (periodic) LSTM vs Static: mean=151.900 vs 151.900, diff=0.000, t=nan, p=nan, d=0.000, n=3 → 비유의
- p99_latency (periodic) LinUCB vs Static: mean=151.900 vs 151.900, diff=0.000, t=nan, p=nan, d=0.000, n=3 → 비유의
- stability_score (periodic) LSTM vs LinUCB: mean=0.325 vs 0.335, diff=-0.010, t=-1.424, p=0.291, d=-0.822, n=3 → 비유의
- stability_score (periodic) LSTM vs Static: mean=0.325 vs 0.390, diff=-0.066, t=-3.736, p=0.065, d=-2.157, n=3 → 비유의
- stability_score (periodic) LinUCB vs Static: mean=0.335 vs 0.390, diff=-0.056, t=-5.281, p=0.034, d=-3.049, n=3 → 유의
- success_rate (failure) LSTM vs LinUCB: mean=1.000 vs 1.000, diff=0.000, t=nan, p=nan, d=0.000, n=3 → 비유의
- success_rate (failure) LSTM vs Static: mean=1.000 vs 1.000, diff=0.000, t=nan, p=nan, d=0.000, n=3 → 비유의
- success_rate (failure) LinUCB vs Static: mean=1.000 vs 1.000, diff=0.000, t=nan, p=nan, d=0.000, n=3 → 비유의
- p99_latency (failure) LSTM vs LinUCB: mean=145.176 vs 145.176, diff=0.000, t=nan, p=nan, d=0.000, n=3 → 비유의
- p99_latency (failure) LSTM vs Static: mean=145.176 vs 145.176, diff=0.000, t=nan, p=nan, d=0.000, n=3 → 비유의
- p99_latency (failure) LinUCB vs Static: mean=145.176 vs 145.176, diff=0.000, t=nan, p=nan, d=0.000, n=3 → 비유의
- stability_score (failure) LSTM vs LinUCB: mean=1.000 vs 1.000, diff=0.000, t=nan, p=nan, d=0.000, n=3 → 비유의
- stability_score (failure) LSTM vs Static: mean=1.000 vs 1.000, diff=0.000, t=nan, p=nan, d=0.000, n=3 → 비유의
- stability_score (failure) LinUCB vs Static: mean=1.000 vs 1.000, diff=0.000, t=nan, p=nan, d=0.000, n=3 → 비유의

## Pairwise Comparisons

| Scenario | Metric | Model A | Mean A | Model B | Mean B | Diff | p-value | Effect size | Significant | n |
|---|---|---|---|---|---|---|---|---|---|---|
| periodic | success_rate | LSTM | 0.597 | LinUCB | 0.749 | -0.152 | 0.002 | -12.914 | Yes | 3 |
| periodic | success_rate | LSTM | 0.597 | Static | 0.867 | -0.270 | 0.000 | -33.239 | Yes | 3 |
| periodic | success_rate | LinUCB | 0.749 | Static | 0.867 | -0.118 | 0.009 | -5.946 | Yes | 3 |
| periodic | p99_latency | LSTM | 151.900 | LinUCB | 151.900 | 0.000 | nan | 0.000 | No | 3 |
| periodic | p99_latency | LSTM | 151.900 | Static | 151.900 | 0.000 | nan | 0.000 | No | 3 |
| periodic | p99_latency | LinUCB | 151.900 | Static | 151.900 | 0.000 | nan | 0.000 | No | 3 |
| periodic | stability_score | LSTM | 0.325 | LinUCB | 0.335 | -0.010 | 0.291 | -0.822 | No | 3 |
| periodic | stability_score | LSTM | 0.325 | Static | 0.390 | -0.066 | 0.065 | -2.157 | No | 3 |
| periodic | stability_score | LinUCB | 0.335 | Static | 0.390 | -0.056 | 0.034 | -3.049 | Yes | 3 |
| failure | success_rate | LSTM | 1.000 | LinUCB | 1.000 | 0.000 | nan | 0.000 | No | 3 |
| failure | success_rate | LSTM | 1.000 | Static | 1.000 | 0.000 | nan | 0.000 | No | 3 |
| failure | success_rate | LinUCB | 1.000 | Static | 1.000 | 0.000 | nan | 0.000 | No | 3 |
| failure | p99_latency | LSTM | 145.176 | LinUCB | 145.176 | 0.000 | nan | 0.000 | No | 3 |
| failure | p99_latency | LSTM | 145.176 | Static | 145.176 | 0.000 | nan | 0.000 | No | 3 |
| failure | p99_latency | LinUCB | 145.176 | Static | 145.176 | 0.000 | nan | 0.000 | No | 3 |
| failure | stability_score | LSTM | 1.000 | LinUCB | 1.000 | 0.000 | nan | 0.000 | No | 3 |
| failure | stability_score | LSTM | 1.000 | Static | 1.000 | 0.000 | nan | 0.000 | No | 3 |
| failure | stability_score | LinUCB | 1.000 | Static | 1.000 | 0.000 | nan | 0.000 | No | 3 |