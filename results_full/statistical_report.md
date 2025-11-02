# 실험 결과 요약

## 시나리오별 성능 요약

### Gradual
```
        success_rate  p99_latency  stability_score  adaptation_time  predictive_mae  tracking_lag_seconds  pattern_recognition  proactive_adjustment
model                                                                                                                                               
LSTM           0.487      255.179            0.367              NaN        2023.506                   NaN                  NaN                   NaN
LSTM           0.473      255.179            0.373              NaN        2023.506                   NaN                  NaN                   NaN
LSTM           0.520      255.179            0.407              NaN        2023.506                   NaN                  NaN                   NaN
LSTM           0.487      255.179            0.413              NaN        2023.506                   NaN                  NaN                   NaN
LSTM           0.453      255.179            0.287              NaN        2023.506                   NaN                  NaN                   NaN
LSTM           0.473      255.179            0.400              NaN        2023.506                   NaN                  NaN                   NaN
LSTM           0.460      255.179            0.400              NaN        2023.506                   NaN                  NaN                   NaN
LSTM           0.513      255.179            0.420              NaN        2023.506                   NaN                  NaN                   NaN
LSTM           0.480      255.179            0.433              NaN        2023.506                   NaN                  NaN                   NaN
LSTM           0.493      255.179            0.327              NaN        2023.506                   NaN                  NaN                   NaN
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
LSTM           0.518      185.958            0.034              NaN         443.234                   NaN                  NaN                   NaN
LSTM           0.533      185.958            0.036              NaN         443.234                   NaN                  NaN                   NaN
LSTM           0.523      185.958            0.038              NaN         443.234                   NaN                  NaN                   NaN
LSTM           0.527      185.958            0.037              NaN         443.234                   NaN                  NaN                   NaN
LSTM           0.518      185.958            0.035              NaN         443.234                   NaN                  NaN                   NaN
LSTM           0.546      185.958            0.037              NaN         443.234                   NaN                  NaN                   NaN
LSTM           0.504      185.958            0.036              NaN         443.234                   NaN                  NaN                   NaN
LSTM           0.534      185.958            0.038              NaN         443.234                   NaN                  NaN                   NaN
LSTM           0.539      185.958            0.040              NaN         443.234                   NaN                  NaN                   NaN
LSTM           0.534      185.958            0.037              NaN         443.234                   NaN                  NaN                   NaN
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
LSTM           0.404      235.758            0.035              NaN        1089.861                   NaN               -0.989                 0.006
LSTM           0.399      235.758            0.041              NaN        1089.861                   NaN               -0.989                 0.006
LSTM           0.404      235.758            0.038              NaN        1089.861                   NaN               -0.989                 0.006
LSTM           0.417      235.758            0.039              NaN        1089.861                   NaN               -0.989                 0.006
LSTM           0.406      235.758            0.037              NaN        1089.861                   NaN               -0.989                 0.006
LSTM           0.422      235.758            0.037              NaN        1089.861                   NaN               -0.989                 0.006
LSTM           0.400      235.758            0.036              NaN        1089.861                   NaN               -0.989                 0.006
LSTM           0.406      235.758            0.039              NaN        1089.861                   NaN               -0.989                 0.006
LSTM           0.400      235.758            0.050              NaN        1089.861                   NaN               -0.989                 0.006
LSTM           0.404      235.758            0.037              NaN        1089.861                   NaN               -0.989                 0.006
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
LSTM           0.578      243.897            0.346           92.000        1726.519                   NaN                  NaN                   NaN
LSTM           0.605      243.897            0.357           90.000        1726.519                   NaN                  NaN                   NaN
LSTM           0.595      243.897            0.351           91.000        1726.519                   NaN                  NaN                   NaN
LSTM           0.578      243.897            0.357           91.000        1726.519                   NaN                  NaN                   NaN
LSTM           0.573      243.897            0.351           90.000        1726.519                   NaN                  NaN                   NaN
LSTM           0.584      243.897            0.357           91.000        1726.519                   NaN                  NaN                   NaN
LSTM           0.595      243.897            0.362           89.000        1726.519                   NaN                  NaN                   NaN
LSTM           0.595      243.897            0.346           96.000        1726.519                   NaN                  NaN                   NaN
LSTM           0.595      243.897            0.357           90.000        1726.519                   NaN                  NaN                   NaN
LSTM           0.589      243.897            0.362           91.000        1726.519                   NaN                  NaN                   NaN
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
- success_rate (normal): t=-37.760, p=0.000, d=-11.941 → 유의