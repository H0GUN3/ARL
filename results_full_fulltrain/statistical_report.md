# 실험 결과 요약

## 시나리오별 성능 요약

### Gradual
```
        success_rate  p99_latency  stability_score  adaptation_time  predictive_mae  tracking_lag_seconds  pattern_recognition  proactive_adjustment
model                                                                                                                                               
LSTM           0.627      255.179            0.393              NaN        1495.483                   NaN                  NaN                   NaN
LSTM           0.627      255.179            0.380              NaN        1495.483                   NaN                  NaN                   NaN
LSTM           0.687      255.179            0.447              NaN        1495.483                   NaN                  NaN                   NaN
LSTM           0.593      255.179            0.433              NaN        1495.483                   NaN                  NaN                   NaN
LSTM           0.613      255.179            0.287              NaN        1495.483                   NaN                  NaN                   NaN
LSTM           0.647      255.179            0.440              NaN        1495.483                   NaN                  NaN                   NaN
LSTM           0.627      255.179            0.420              NaN        1495.483                   NaN                  NaN                   NaN
LSTM           0.640      255.179            0.453              NaN        1495.483                   NaN                  NaN                   NaN
LSTM           0.673      255.179            0.453              NaN        1495.483                   NaN                  NaN                   NaN
LSTM           0.587      255.179            0.327              NaN        1495.483                   NaN                  NaN                   NaN
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
LSTM           1.000      185.958            1.000              NaN         144.415                   NaN                  NaN                   NaN
LSTM           1.000      185.958            1.000              NaN         144.415                   NaN                  NaN                   NaN
LSTM           1.000      185.958            1.000              NaN         144.415                   NaN                  NaN                   NaN
LSTM           1.000      185.958            1.000              NaN         144.415                   NaN                  NaN                   NaN
LSTM           1.000      185.958            1.000              NaN         144.415                   NaN                  NaN                   NaN
LSTM           1.000      185.958            1.000              NaN         144.415                   NaN                  NaN                   NaN
LSTM           0.999      185.958            1.000              NaN         144.415                   NaN                  NaN                   NaN
LSTM           1.000      185.958            1.000              NaN         144.415                   NaN                  NaN                   NaN
LSTM           1.000      185.958            1.000              NaN         144.415                   NaN                  NaN                   NaN
LSTM           1.000      185.958            1.000              NaN         144.415                   NaN                  NaN                   NaN
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
LSTM           0.718      235.758            0.447              NaN         768.879                   NaN               -0.989                 0.006
LSTM           0.741      235.758            0.452              NaN         768.879                   NaN               -0.989                 0.006
LSTM           0.737      235.758            0.446              NaN         768.879                   NaN               -0.989                 0.006
LSTM           0.736      235.758            0.450              NaN         768.879                   NaN               -0.989                 0.006
LSTM           0.722      235.758            0.452              NaN         768.879                   NaN               -0.989                 0.006
LSTM           0.744      235.758            0.452              NaN         768.879                   NaN               -0.989                 0.006
LSTM           0.735      235.758            0.454              NaN         768.879                   NaN               -0.989                 0.006
LSTM           0.735      235.758            0.451              NaN         768.879                   NaN               -0.989                 0.006
LSTM           0.740      235.758            0.453              NaN         768.879                   NaN               -0.989                 0.006
LSTM           0.732      235.758            0.448              NaN         768.879                   NaN               -0.989                 0.006
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
LSTM           0.681      243.897            0.351           89.000        1524.054                   NaN                  NaN                   NaN
LSTM           0.703      243.897            0.362           82.000        1524.054                   NaN                  NaN                   NaN
LSTM           0.686      243.897            0.368           84.000        1524.054                   NaN                  NaN                   NaN
LSTM           0.703      243.897            0.362           81.000        1524.054                   NaN                  NaN                   NaN
LSTM           0.681      243.897            0.357           81.000        1524.054                   NaN                  NaN                   NaN
LSTM           0.692      243.897            0.368           89.000        1524.054                   NaN                  NaN                   NaN
LSTM           0.703      243.897            0.368           89.000        1524.054                   NaN                  NaN                   NaN
LSTM           0.697      243.897            0.351           89.000        1524.054                   NaN                  NaN                   NaN
LSTM           0.741      243.897            0.389           81.000        1524.054                   NaN                  NaN                   NaN
LSTM           0.697      243.897            0.368           84.000        1524.054                   NaN                  NaN                   NaN
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
- success_rate (normal): t=130.560, p=0.000, d=41.287 → 유의