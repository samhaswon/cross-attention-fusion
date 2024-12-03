# cross-attention-fusion

models:

mhsa_vit: Multi-Head Self-Attention, concatenation

mhsa_2: Multi-Head Self-Attention, addition

early_concat: early concatenation

## Accuracies

|              |           Train            |            Eval            |
|:------------:|:--------------------------:|:--------------------------:|
|   mhsa_vit   |          74.9665%          |          63.2500%          |
|    mhsa_2    |          dnf/nan           |          dnf/nan           |
| early_concat | 99.7389% (max of 99.7570%) | 80.2500% (max of 82.0000%) |