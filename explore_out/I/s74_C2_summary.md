S74 pilot, configuration C2: ICMC S=2 V=4 NB=40 block words 3..7 M=200 words/source L=480; NOMIX ICMC S=1 V=4 NB=80 block words 3..7 M=400 words/source L=480
code SHA fa23cf4c1e13, CPU Intel(R) Xeon(R) Processor @ 2.10GHz

| arm | seed | SET end | SET 1 / 2 / 3-4 / 5+ | loss end | probe L1/L2/L3/final | minutes |
|---|---|---|---|---|---|---|
| ORACLE | 800 | 0.916 | 0.917 / 0.911 / 0.914 / 0.924 | 1.144 | 0.50 / 0.50 / 0.48 / 0.48 | 166 |
| ORACLE | 801 | 0.917 | 0.914 / 0.912 / 0.916 / 0.925 | 1.143 | 0.50 / 0.50 / 0.49 / 0.47 | 166 |
| ORACLE | 802 | 0.917 | 0.914 / 0.913 / 0.916 / 0.924 | 1.145 | 0.50 / 0.51 / 0.50 / 0.48 | 164 |
| ORACLE | 803 | 0.918 | 0.919 / 0.913 / 0.915 / 0.925 | 1.142 | 0.50 / 0.49 / 0.50 / 0.49 | 164 |
| SINGLE | 800 | 0.812 | 0.707 / 0.829 / 0.838 / 0.844 | 1.213 | 0.50 / 0.87 / 0.98 / 0.98 | 155 |
| SINGLE | 801 | 0.812 | 0.712 / 0.830 / 0.838 / 0.841 | 1.216 | 0.50 / 0.87 / 0.98 / 0.98 | 155 |
| SINGLE | 802 | 0.654 | 0.603 / 0.651 / 0.673 / 0.671 | 1.296 | 0.50 / 0.52 / 0.51 / 0.51 | 155 |
| SINGLE | 803 | 0.863 | 0.787 / 0.874 / 0.883 / 0.886 | 1.185 | 0.50 / 0.91 / 0.98 / 0.96 | 157 |

USABLE (ORACLE >= 0.9 and SINGLE <= ORACLE - 0.2 on >= 3/4 seeds): 1/4 seeds -> NOT USABLE
  ORACLE: SET median 0.917 (range 0.916-0.918); wall time per run median 165 min
  SINGLE: SET median 0.812 (range 0.654-0.863); wall time per run median 155 min
