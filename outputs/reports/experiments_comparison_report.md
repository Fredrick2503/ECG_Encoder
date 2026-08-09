# Experiments Comparison Report — MasterMind Loop v1

**Generated:** 2026-08-09 19:04

## Summary Table

| Trial | Arch | Filter | Balance | Loss | AUC | F1 | Acc | Sens | Spec |
|---|---|---|---|---|---|---|---|---|---|
| T24_cb_loss | transformer | bandpass_notch | none | cb_loss | 0.7888 | 0.4920 | 0.3600 | 0.7332 | 0.6750 |
| T25_ldam | transformer | bandpass_notch | none | ldam | 0.8036 | 0.4744 | 0.4267 | 0.5506 | 0.7955 |
| T26_resnet_cb_loss | resnet_se | full_stack | none | cb_loss | 0.8550 | 0.5953 | 0.4733 | 0.6522 | 0.8686 |
| T27_resnet_bal_cb | resnet_se | bandpass_notch | average | cb_loss | 0.7946 | 0.5781 | 0.3000 | 0.6844 | 0.7573 |
| T28_high_dropout | transformer | bandpass_notch | none | asl | 0.7996 | 0.5093 | 0.4000 | 0.6487 | 0.7689 |
| T29_heavy_wd | transformer | bandpass_notch | none | asl | 0.8084 | 0.5327 | 0.4200 | 0.6518 | 0.7833 |

## Best Model

**Trial:** T26_resnet_cb_loss  
**Architecture:** resnet_se  
**Filter:** full_stack  
**Balance:** none  
**Loss:** cb_loss  
**Macro AUC:** 0.8550  
**Macro F1:** 0.5953  
