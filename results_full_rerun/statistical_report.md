# 실험 결과 요약

## 시나리오별 성능 요약

### Gradual
```
        success_rate  p99_latency  stability_score  adaptation_time  predictive_mae  tracking_lag_seconds  pattern_recognition  proactive_adjustment
model                                                                                                                                               
LSTM           0.500      255.179            0.367              NaN        1993.896                   NaN                  NaN                   NaN
LSTM           0.473      255.179            0.373              NaN        1993.896                   NaN                  NaN                   NaN
LSTM           0.527      255.179            0.407              NaN        1993.896                   NaN                  NaN                   NaN
LSTM           0.493      255.179            0.413              NaN        1993.896                   NaN                  NaN                   NaN
LSTM           0.460      255.179            0.287              NaN        1993.896                   NaN                  NaN                   NaN
LSTM           0.493      255.179            0.400              NaN        1993.896                   NaN                  NaN                   NaN
LSTM           0.473      255.179            0.400              NaN        1993.896                   NaN                  NaN                   NaN
LSTM           0.513      255.179            0.420              NaN        1993.896                   NaN                  NaN                   NaN
LSTM           0.507      255.179            0.433              NaN        1993.896                   NaN                  NaN                   NaN
LSTM           0.493      255.179            0.327              NaN        1993.896                   NaN                  NaN                   NaN
LinUCB         0.447      255.179            0.287              NaN             NaN               116.000                  NaN                   NaN
LinUCB         0.460      255.179            0.253              NaN             NaN               116.000                  NaN                   NaN
LinUCB         0.487      255.179            0.327              NaN             NaN               116.000                  NaN                   NaN
LinUCB         0.460      255.179            0.320              NaN             NaN               116.000                  NaN                   NaN
LinUCB         0.413      255.179            0.247              NaN             NaN               116.000                  NaN                   NaN
LinUCB         0.467      255.179            0.313              NaN             NaN               116.000                  NaN                   NaN
LinUCB         0.440      255.179            0.313              NaN             NaN               116.000                  NaN                   NaN
LinUCB         0.460      255.179            0.273              NaN             NaN               116.000                  NaN                   NaN
LinUCB         0.473      255.179            0.313              NaN             NaN               116.000                  NaN                   NaN
LinUCB         0.440      255.179            0.273              NaN             NaN               116.000                  NaN                   NaN
Static         0.453      255.179            0.287              NaN             NaN                   NaN                  NaN                   NaN
Static         0.467      255.179            0.253              NaN             NaN                   NaN                  NaN                   NaN
Static         0.493      255.179            0.333              NaN             NaN                   NaN                  NaN                   NaN
Static         0.460      255.179            0.320              NaN             NaN                   NaN                  NaN                   NaN
Static         0.420      255.179            0.280              NaN             NaN                   NaN                  NaN                   NaN
Static         0.487      255.179            0.313              NaN             NaN                   NaN                  NaN                   NaN
Static         0.447      255.179            0.313              NaN             NaN                   NaN                  NaN                   NaN
Static         0.467      255.179            0.280              NaN             NaN                   NaN                  NaN                   NaN
Static         0.487      255.179            0.313              NaN             NaN                   NaN                  NaN                   NaN
Static         0.453      255.179            0.273              NaN             NaN                   NaN                  NaN                   NaN
```

### Normal
```
        success_rate  p99_latency  stability_score  adaptation_time  predictive_mae  tracking_lag_seconds  pattern_recognition  proactive_adjustment
model                                                                                                                                               
LSTM           0.562      185.958            0.035              NaN         407.957                   NaN                  NaN                   NaN
LSTM           0.576      185.958            0.041              NaN         407.957                   NaN                  NaN                   NaN
LSTM           0.568      185.958            0.038              NaN         407.957                   NaN                  NaN                   NaN
LSTM           0.571      185.958            0.038              NaN         407.957                   NaN                  NaN                   NaN
LSTM           0.564      185.958            0.037              NaN         407.957                   NaN                  NaN                   NaN
LSTM           0.594      185.958            0.037              NaN         407.957                   NaN                  NaN                   NaN
LSTM           0.556      185.958            0.036              NaN         407.957                   NaN                  NaN                   NaN
LSTM           0.586      185.958            0.038              NaN         407.957                   NaN                  NaN                   NaN
LSTM           0.592      185.958            0.045              NaN         407.957                   NaN                  NaN                   NaN
LSTM           0.576      185.958            0.037              NaN         407.957                   NaN                  NaN                   NaN
LinUCB         0.599      185.958            0.000              NaN             NaN              1800.000                  NaN                   NaN
LinUCB         0.608      185.958            0.001              NaN             NaN              1800.000                  NaN                   NaN
LinUCB         0.602      185.958            0.001              NaN             NaN              1800.000                  NaN                   NaN
LinUCB         0.611      185.958            0.001              NaN             NaN              1800.000                  NaN                   NaN
LinUCB         0.605      185.958            0.000              NaN             NaN              1800.000                  NaN                   NaN
LinUCB         0.628      185.958            0.000              NaN             NaN              1800.000                  NaN                   NaN
LinUCB         0.601      185.958            0.002              NaN             NaN              1800.000                  NaN                   NaN
LinUCB         0.614      185.958            0.000              NaN             NaN              1800.000                  NaN                   NaN
LinUCB         0.622      185.958            0.001              NaN             NaN              1800.000                  NaN                   NaN
LinUCB         0.605      185.958            0.000              NaN             NaN              1800.000                  NaN                   NaN
Static         0.626      185.958            0.002              NaN             NaN                   NaN                  NaN                   NaN
Static         0.633      185.958            0.001              NaN             NaN                   NaN                  NaN                   NaN
Static         0.625      185.958            0.001              NaN             NaN                   NaN                  NaN                   NaN
Static         0.637      185.958            0.001              NaN             NaN                   NaN                  NaN                   NaN
Static         0.628      185.958            0.000              NaN             NaN                   NaN                  NaN                   NaN
Static         0.652      185.958            0.000              NaN             NaN                   NaN                  NaN                   NaN
Static         0.626      185.958            0.002              NaN             NaN                   NaN                  NaN                   NaN
Static         0.640      185.958            0.001              NaN             NaN                   NaN                  NaN                   NaN
Static         0.643      185.958            0.001              NaN             NaN                   NaN                  NaN                   NaN
Static         0.633      185.958            0.000              NaN             NaN                   NaN                  NaN                   NaN
```

### Periodic
```
        success_rate  p99_latency  stability_score  adaptation_time  predictive_mae  tracking_lag_seconds  pattern_recognition  proactive_adjustment
model                                                                                                                                               
LSTM           0.439      235.758            0.037              NaN        1056.259                   NaN               -0.989                 0.006
LSTM           0.432      235.758            0.041              NaN        1056.259                   NaN               -0.989                 0.006
LSTM           0.444      235.758            0.042              NaN        1056.259                   NaN               -0.989                 0.006
LSTM           0.454      235.758            0.041              NaN        1056.259                   NaN               -0.989                 0.006
LSTM           0.446      235.758            0.037              NaN        1056.259                   NaN               -0.989                 0.006
LSTM           0.459      235.758            0.037              NaN        1056.259                   NaN               -0.989                 0.006
LSTM           0.430      235.758            0.036              NaN        1056.259                   NaN               -0.989                 0.006
LSTM           0.456      235.758            0.039              NaN        1056.259                   NaN               -0.989                 0.006
LSTM           0.441      235.758            0.051              NaN        1056.259                   NaN               -0.989                 0.006
LSTM           0.443      235.758            0.037              NaN        1056.259                   NaN               -0.989                 0.006
LinUCB         0.461      235.758            0.003              NaN             NaN              1800.000               -0.989                 0.006
LinUCB         0.461      235.758            0.001              NaN             NaN              1800.000               -0.989                 0.006
LinUCB         0.484      235.758            0.001              NaN             NaN              1800.000               -0.989                 0.006
LinUCB         0.486      235.758            0.001              NaN             NaN              1800.000               -0.989                 0.006
LinUCB         0.468      235.758            0.000              NaN             NaN              1800.000               -0.989                 0.006
LinUCB         0.489      235.758            0.000              NaN             NaN              1800.000               -0.989                 0.006
LinUCB         0.463      235.758            0.002              NaN             NaN              1800.000               -0.989                 0.006
LinUCB         0.477      235.758            0.001              NaN             NaN              1800.000               -0.989                 0.006
LinUCB         0.462      235.758            0.001              NaN             NaN              1800.000               -0.989                 0.006
LinUCB         0.476      235.758            0.000              NaN             NaN              1800.000               -0.989                 0.006
Static         0.475      235.758            0.003              NaN             NaN                   NaN               -0.989                 0.006
Static         0.482      235.758            0.001              NaN             NaN                   NaN               -0.989                 0.006
Static         0.499      235.758            0.001              NaN             NaN                   NaN               -0.989                 0.006
Static         0.502      235.758            0.001              NaN             NaN                   NaN               -0.989                 0.006
Static         0.487      235.758            0.000              NaN             NaN                   NaN               -0.989                 0.006
Static         0.512      235.758            0.000              NaN             NaN                   NaN               -0.989                 0.006
Static         0.478      235.758            0.002              NaN             NaN                   NaN               -0.989                 0.006
Static         0.497      235.758            0.001              NaN             NaN                   NaN               -0.989                 0.006
Static         0.479      235.758            0.001              NaN             NaN                   NaN               -0.989                 0.006
Static         0.500      235.758            0.000              NaN             NaN                   NaN               -0.989                 0.006
```

### Spike
```
        success_rate  p99_latency  stability_score  adaptation_time  predictive_mae  tracking_lag_seconds  pattern_recognition  proactive_adjustment
model                                                                                                                                               
LSTM           0.578      243.897            0.346           91.000        1703.219                   NaN                  NaN                   NaN
LSTM           0.600      243.897            0.351           86.000        1703.219                   NaN                  NaN                   NaN
LSTM           0.611      243.897            0.357           87.000        1703.219                   NaN                  NaN                   NaN
LSTM           0.595      243.897            0.362           90.000        1703.219                   NaN                  NaN                   NaN
LSTM           0.584      243.897            0.351           90.000        1703.219                   NaN                  NaN                   NaN
LSTM           0.600      243.897            0.362           90.000        1703.219                   NaN                  NaN                   NaN
LSTM           0.600      243.897            0.341           90.000        1703.219                   NaN                  NaN                   NaN
LSTM           0.600      243.897            0.351           91.000        1703.219                   NaN                  NaN                   NaN
LSTM           0.605      243.897            0.357           91.000        1703.219                   NaN                  NaN                   NaN
LSTM           0.600      243.897            0.351           90.000        1703.219                   NaN                  NaN                   NaN
LinUCB         0.605      243.897            0.362           91.000             NaN                90.000                  NaN                   NaN
LinUCB         0.611      243.897            0.357           88.000             NaN                90.000                  NaN                   NaN
LinUCB         0.616      243.897            0.362           89.000             NaN                90.000                  NaN                   NaN
LinUCB         0.600      243.897            0.362           90.000             NaN                90.000                  NaN                   NaN
LinUCB         0.578      243.897            0.351           89.000             NaN                90.000                  NaN                   NaN
LinUCB         0.595      243.897            0.346           89.000             NaN                90.000                  NaN                   NaN
LinUCB         0.605      243.897            0.351           89.000             NaN                90.000                  NaN                   NaN
LinUCB         0.605      243.897            0.346           89.000             NaN                90.000                  NaN                   NaN
LinUCB         0.627      243.897            0.351           89.000             NaN                90.000                  NaN                   NaN
LinUCB         0.605      243.897            0.351           91.000             NaN                90.000                  NaN                   NaN
Static         0.605      243.897            0.362           91.000             NaN                   NaN                  NaN                   NaN
Static         0.611      243.897            0.357           88.000             NaN                   NaN                  NaN                   NaN
Static         0.616      243.897            0.362           89.000             NaN                   NaN                  NaN                   NaN
Static         0.605      243.897            0.362           90.000             NaN                   NaN                  NaN                   NaN
Static         0.584      243.897            0.351           89.000             NaN                   NaN                  NaN                   NaN
Static         0.600      243.897            0.346           89.000             NaN                   NaN                  NaN                   NaN
Static         0.605      243.897            0.351           89.000             NaN                   NaN                  NaN                   NaN
Static         0.611      243.897            0.346           89.000             NaN                   NaN                  NaN                   NaN
Static         0.627      243.897            0.351           89.000             NaN                   NaN                  NaN                   NaN
Static         0.622      243.897            0.351           91.000             NaN                   NaN                  NaN                   NaN
```

## 통계 검정 결과
- success_rate (normal): t=-20.239, p=0.000, d=-6.400 → 유의