Same pumps, same faults, same operating context; only the instrumentation differs (simulator rebuilt with identical seeds). Composite counts only events the instruments can reveal, so the share of all faults caught with the correct type (table 2) is the fairer comparison of what each sensor level buys.

| sensor level   |   pumps |   events (observable) |   unobservable (not scored) |   composite |   recall |   fault-type recall |   false+excess /asset-day |   B1 composite |
|:---------------|--------:|----------------------:|----------------------------:|------------:|---------:|--------------------:|--------------------------:|---------------:|
| T0_LEGACY      |      75 |                   252 |                         110 |        85.2 |     0.89 |                0.8  |                      0.08 |           55.6 |
| T1_RETROFIT    |      75 |                   350 |                          12 |        87.5 |     0.91 |                0.83 |                      0.09 |           54.4 |
| T2_FULL        |      75 |                   358 |                           4 |        91.7 |     0.94 |                0.89 |                      0.06 |           57.5 |

Share of ALL simulated faults (observable or not) detected with the correct fault type:

| fault_type                |   T0_LEGACY |   T1_RETROFIT |   T2_FULL |
|:--------------------------|------------:|--------------:|----------:|
| AIR_ENTRAINMENT           |        0.8  |          0.9  |      0.85 |
| BEARING_WEAR              |        0.08 |          0.81 |      0.85 |
| CAVITATION                |        0.81 |          0.85 |      0.93 |
| COMMUNICATION_LOSS        |        1    |          1    |      1    |
| DISCHARGE_RESTRICTION     |        0.87 |          0.82 |      0.97 |
| IMBALANCE                 |        0    |          0.71 |      1    |
| IMPELLER_WEAR             |        0.81 |          0.88 |      0.94 |
| MISALIGNMENT              |        0.07 |          0.86 |      0.96 |
| SENSOR_CALIBRATION_SHIFT  |        0.18 |          0.32 |      0.32 |
| SENSOR_DRIFT              |        0.32 |          0.74 |      0.88 |
| SENSOR_NOISE_SPIKES       |        0.61 |          0.74 |      0.94 |
| SENSOR_STUCK              |        0.67 |          0.94 |      0.97 |
| SUCTION_STRAINER_CLOGGING |        0.82 |          0.97 |      0.94 |