# Corruption Severity Benchmark Report

This report details the model robustness across 5 severity levels of signal corruption.

## 1. Failure-Mode Impact Ranking (F1 Drop from Clean to Level 5)

| Corruption             |   F1 at Level 0 (Clean) |   F1 at Level 5 (Severe) |   Macro F1 Drop |
|:-----------------------|------------------------:|-------------------------:|----------------:|
| Baseline Offset        |                0.722079 |                 0.126559 |        0.595521 |
| Baseline Wander        |                0.722079 |                 0.19028  |        0.531799 |
| Amplitude Scaling      |                0.722079 |                 0.207413 |        0.514667 |
| High-Frequency Noise   |                0.722079 |                 0.372335 |        0.349744 |
| EMG Noise              |                0.722079 |                 0.54507  |        0.177009 |
| Powerline Interference |                0.722079 |                 0.722079 |        0        |

## 2. Detailed Performance Table

| Corruption             |   Severity |   Macro F1 |   Macro AUC |   Macro ECE |   Brier Score |
|:-----------------------|-----------:|-----------:|------------:|------------:|--------------:|
| Baseline Wander        |          0 |   0.722079 |    0.909252 |   0.0487309 |     0.084846  |
| Powerline Interference |          0 |   0.722079 |    0.909252 |   0.0487309 |     0.084846  |
| High-Frequency Noise   |          0 |   0.722079 |    0.909252 |   0.0487309 |     0.084846  |
| Amplitude Scaling      |          0 |   0.722079 |    0.909252 |   0.0487309 |     0.084846  |
| Baseline Offset        |          0 |   0.722079 |    0.909252 |   0.0487309 |     0.084846  |
| EMG Noise              |          0 |   0.722079 |    0.909252 |   0.0487309 |     0.084846  |
| Baseline Wander        |          1 |   0.7249   |    0.910114 |   0.0459136 |     0.0848112 |
| Baseline Wander        |          2 |   0.709345 |    0.909204 |   0.0552266 |     0.0864022 |
| Baseline Wander        |          3 |   0.687988 |    0.904035 |   0.0554943 |     0.0911746 |
| Baseline Wander        |          4 |   0.601371 |    0.869411 |   0.0908753 |     0.118536  |
| Baseline Wander        |          5 |   0.19028  |    0.752189 |   0.144161  |     0.178575  |
| Powerline Interference |          1 |   0.722079 |    0.909252 |   0.0487309 |     0.084846  |
| Powerline Interference |          2 |   0.722079 |    0.909252 |   0.0487309 |     0.084846  |
| Powerline Interference |          3 |   0.722079 |    0.909252 |   0.0487309 |     0.084846  |
| Powerline Interference |          4 |   0.722079 |    0.909252 |   0.0487309 |     0.084846  |
| Powerline Interference |          5 |   0.722079 |    0.909252 |   0.0487309 |     0.084846  |
| High-Frequency Noise   |          1 |   0.724353 |    0.909514 |   0.0471486 |     0.084867  |
| High-Frequency Noise   |          2 |   0.720126 |    0.909532 |   0.0490503 |     0.0849019 |
| High-Frequency Noise   |          3 |   0.700036 |    0.905669 |   0.0441608 |     0.0892992 |
| High-Frequency Noise   |          4 |   0.488529 |    0.873218 |   0.120818  |     0.140277  |
| High-Frequency Noise   |          5 |   0.372335 |    0.78685  |   0.149405  |     0.181483  |
| Amplitude Scaling      |          1 |   0.696689 |    0.901115 |   0.0598242 |     0.0882561 |
| Amplitude Scaling      |          2 |   0.622422 |    0.880123 |   0.0636133 |     0.0981076 |
| Amplitude Scaling      |          3 |   0.553946 |    0.852811 |   0.0774906 |     0.112938  |
| Amplitude Scaling      |          4 |   0.401956 |    0.800037 |   0.139677  |     0.152928  |
| Amplitude Scaling      |          5 |   0.207413 |    0.689856 |   0.193787  |     0.200103  |
| Baseline Offset        |          1 |   0.707488 |    0.896303 |   0.0738912 |     0.0970057 |
| Baseline Offset        |          2 |   0.656395 |    0.880665 |   0.100849  |     0.112334  |
| Baseline Offset        |          3 |   0.441518 |    0.813503 |   0.128387  |     0.148746  |
| Baseline Offset        |          4 |   0.24076  |    0.740597 |   0.145705  |     0.18023   |
| Baseline Offset        |          5 |   0.126559 |    0.736777 |   0.192163  |     0.206292  |
| EMG Noise              |          1 |   0.723836 |    0.909374 |   0.0490152 |     0.0847889 |
| EMG Noise              |          2 |   0.723551 |    0.909185 |   0.0441234 |     0.0849012 |
| EMG Noise              |          3 |   0.724467 |    0.909024 |   0.0485689 |     0.0858871 |
| EMG Noise              |          4 |   0.689134 |    0.905391 |   0.0499337 |     0.0914285 |
| EMG Noise              |          5 |   0.54507  |    0.883293 |   0.108252  |     0.129517  |

## 3. Visualized Curves
The plots showing F1, AUC, ECE, and Brier score against severity are saved to `outputs/figures/severity_robustness_curves.png`.
