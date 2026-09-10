# ECG Biomarker Extraction Summary Report

## Execution Summary
- **Total ECG Records Found**: 21837
- **Successfully Extracted Records**: 21808 (99.87%)
- **Failed Records**: 29 (0.13%)
- **Records with Quality/Validation Warnings**: 4556
- **Total Execution Time**: 2777.55 seconds
- **Average Speed**: 7.85 records/second

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
| abnormal_qtc_interval | 2432 |
| abnormal_pr_interval | 2244 |
| abnormal_qrs_duration | 604 |
| abnormal_heart_rate | 45 |
| II_delineation_error: cannot convert float NaN to integer | 13 |
| V5_delineation_error: cannot convert float NaN to integer | 13 |
| V1_delineation_error: cannot convert float NaN to integer | 13 |
| I_delineation_error: cannot convert float NaN to integer | 13 |
| failed_pr_interval | 9 |
| failed_qrs_duration | 1 |

## Suspicious or Extreme Values
- heart_rate: 21 records have values < 30 or > 200 (Min: 13.5, Max: 29.9)
- qtc_interval: 166 records have values < 250ms or > 600ms (Min: 197.9, Max: 719.5)
- qrs_duration: 1 records have values < 50ms or > 200ms (Min: 46.9, Max: 46.9)

## Biomarker Statistical Summary
| Biomarker Feature | Min Value | Max Value | Missing % |
| --- | --- | --- | --- |
| heart_rate | 13.5257 | 190.5388 | 0.00% |
| mean_rr | 314.8966 | 4436.0000 | 0.00% |
| sd_rr | 0.0000 | 3542.0000 | 0.00% |
| p_amplitude | -1.3041 | 1.1859 | 2.16% |
| p_duration | 52.0000 | 198.0000 | 2.57% |
| pr_interval | 50.0000 | 280.0000 | 17.36% |
| v1_r_amplitude | -4.3326 | 0.9428 | 0.00% |
| v1_s_amplitude | -5.1495 | 0.4459 | 0.00% |
| v5_r_amplitude | -1.6795 | 4.5577 | 0.00% |
| max_r_v1_v6 | -0.7128 | 5.8411 | 0.00% |
| r_progression_slope | -0.3172 | 1.4388 | 0.00% |
| max_st_elevation | 0.0016 | 7.0998 | 0.00% |
| max_st_depression | 0.0024 | 3.6176 | 0.00% |
| num_leads_st_deviation | 0.0000 | 12.0000 | 0.00% |
| max_t_amplitude | 0.0587 | 8.2816 | 0.00% |
| mean_t_amplitude | 0.0277 | 2.6216 | 0.00% |
| num_leads_t_inversion | 1.0000 | 11.0000 | 0.00% |
| qrs_duration | 46.9286 | 171.7000 | 0.33% |
| qt_interval | 200.0000 | 600.0000 | 13.97% |
| qtc_interval | 197.9211 | 719.4894 | 13.97% |
| qrs_axis | -179.6205 | 176.4873 | 0.00% |
| t_wave_axis | -179.9999 | 179.9516 | 0.00% |
| qrs_t_angle | 0.0014 | 179.9928 | 0.00% |
| sokolow_lyon | -1.4248 | 6.8208 | 0.00% |
