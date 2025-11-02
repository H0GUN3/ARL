# BurstGPT Dataset EDA

This report summarizes the exploratory analysis executed via `eda.py`.

## Dataset Overview

- Total rows: 5,288,173
- Unique models: 2
- Time span (seconds): 5.0 → 10454395.0
- Source files: BurstGPT_1.csv, BurstGPT_2.csv

## Missing Values (%)

```
             field  missing_pct
0        timestamp         0.00
1            model         0.00
2   request_tokens         0.00
3  response_tokens         0.00
4     total_tokens         0.00
5         log_type         0.00
6      source_file         0.00
7       tokens_gap         0.00
8   response_ratio         1.88
```

## Log Type Distribution

```
                     rows
log_type                 
API log           5052894
Conversation log   235279
```

## Token Distribution

```
       request_tokens  response_tokens  total_tokens  tokens_gap  response_ratio
count      5288173.00       5288173.00    5288173.00  5288173.00      5188525.00
mean           352.45            58.94        411.40        0.00            0.11
std            482.56           171.90        565.04        0.00            0.14
min              0.00             0.00          0.00        0.00            0.00
50%            217.00             7.00        229.00        0.00            0.03
90%            599.00           147.00        767.00        0.00            0.28
99%           2953.00           716.00       3152.00        0.00            0.68
max          31407.00         12472.00      31924.00        0.00            0.99
```

## Response Ratio Outliers (top 10)

No extreme ratios detected.

## Token Correlations

```
                 request_tokens  response_tokens  total_tokens
request_tokens             1.00             0.34          0.96
response_tokens            0.34             1.00          0.60
total_tokens               0.96             0.60          1.00
```

## Activity per Model

```
        request_tokens               response_tokens               total_tokens              
                 count   mean median           count   mean median        count   mean median
model                                                                                        
ChatGPT        4970268 332.98 215.00         4970268  47.11   7.00      4970268 380.09 227.00
GPT-4           317905 656.92 466.00          317905 243.97 102.00       317905 900.89 648.00
```

## Request Volume per Second

```
       requests_per_second
count           2276653.00
mean                  2.32
std                   2.21
min                   1.00
50%                   2.00
90%                   5.00
99%                  11.00
max                  91.00
```

## Request Volume Distribution (sample)

```
                     seconds_with_count
requests_per_second                    
1                               1054339
2                                586083
3                                240693
4                                165154
5                                 80287
6                                 59241
7                                 26286
8                                 18169
9                                  9793
10                                11323
11                                 5936
12                                 4427
13                                 1951
14                                 1579
15                                  982
16                                 1189
17                                  641
18                                  476
19                                  487
20                                 5671
```

## Highest Request Seconds (top 10)

```
      second  request_count
0 4449998.00             91
1 4450914.00             82
2 4450087.00             79
3 4449864.00             75
4 4449877.00             72
5 4450061.00             72
6 4449950.00             71
7 4449857.00             69
8 4449870.00             69
9 4449974.00             69
```
