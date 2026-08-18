# ECG Biomarker Extraction Summary Report

## Execution Summary
- **Total ECG Records Found**: 21837
- **Successfully Extracted Records**: 21808 (99.87%)
- **Failed Records**: 29 (0.13%)
- **Records with Quality/Validation Warnings**: 19683
- **Total Execution Time**: 2883.20 seconds
- **Average Speed**: 7.56 records/second

## Final Suitability Verdict
> [!NOTE]
> SUITABLE: The extraction success rate is high and overall missing feature rate is low. Missing values can be safely handled using standard imputation methods.

## Class-Label Counts
| Diagnostic Class | Count |
| --- | --- |
| NORM | 9528 |
| MI | 5477 |
| STTC | 5123 |
| CD | 4893 |
| HYP | 2652 |

## Most Common Quality/Validation Warnings
| Warning Type | Count |
| --- | --- |
| abnormal_qrs_duration | 18169 |
| abnormal_pr_interval | 11173 |
| abnormal_qtc_interval | 6311 |
| failed_pr_interval | 276 |
| abnormal_heart_rate | 45 |
| II_delineation_error: cannot convert float NaN to integer | 13 |
| V5_delineation_error: cannot convert float NaN to integer | 13 |
| V1_delineation_error: cannot convert float NaN to integer | 13 |
| I_delineation_error: cannot convert float NaN to integer | 13 |
| failed_qrs_duration | 5 |

## Suspicious or Extreme Values
- heart_rate: 21 records have values < 30 or > 200 (Min: 13.5, Max: 29.9)
- qtc_interval: 618 records have values < 250ms or > 600ms (Min: 155.7, Max: 819.5)
- qrs_duration: 3856 records have values < 50ms or > 200ms (Min: 200.1, Max: 349.3)

## Biomarker Statistical Summary
| Biomarker Feature | Min Value | Max Value | Missing % |
| --- | --- | --- | --- |
| heart_rate | 13.5257 | 190.5388 | 0.00% |
| mean_rr | 314.8966 | 4436.0000 | 0.00% |
| sd_rr | 0.0000 | 3542.0000 | 0.00% |
| p_amplitude | -0.9859 | 1.3008 | 0.06% |
| p_duration | 33.6250 | 275.6000 | 0.06% |
| pr_interval | 50.0000 | 391.7500 | 1.52% |
| v1_r_amplitude | -4.3552 | 1.0856 | 0.00% |
| v1_s_amplitude | -5.1721 | 0.6190 | 0.00% |
| v5_r_amplitude | -1.6401 | 3.9957 | 0.00% |
| max_r_v1_v6 | -0.6311 | 5.8411 | 0.00% |
| r_progression_slope | -0.3690 | 1.4431 | 0.00% |
| max_st_elevation | 0.0026 | 7.6292 | 0.00% |
| max_st_depression | 0.0041 | 5.5512 | 0.00% |
| num_leads_st_deviation | 0.0000 | 12.0000 | 0.00% |
| max_t_amplitude | 0.0293 | 8.2816 | 0.00% |
| mean_t_amplitude | 0.0151 | 2.6062 | 0.00% |
| num_leads_t_inversion | 1.0000 | 11.0000 | 0.00% |
| qrs_duration | 62.5714 | 349.3333 | 0.10% |
| qt_interval | 146.1818 | 885.0000 | 0.58% |
| qtc_interval | 155.7369 | 819.5059 | 0.59% |
| qrs_axis | -179.2007 | 176.7217 | 0.00% |
| t_wave_axis | -179.9999 | 179.9564 | 0.00% |
| qrs_t_angle | 0.0008 | 179.9798 | 0.00% |
| sokolow_lyon | -1.3115 | 6.8565 | 0.00% |
