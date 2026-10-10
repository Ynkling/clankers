S74 pilot, configuration C1: ICMC S=2 V=4 NB=16 block words 6..14 M=176 words/source L=384; NOMIX ICMC S=1 V=4 NB=32 block words 6..14 M=352 words/source L=384
code SHA 49c699613b42, CPU Intel(R) Xeon(R) Processor @ 2.10GHz

| arm | seed | SET end | SET 1 / 2 / 3-4 / 5+ | loss end | probe L1/L2/L3/final | minutes |
|---|---|---|---|---|---|---|
| ORACLE | 800 | 0.918 | 0.932 / 0.898 / 0.901 / 0.924 | 1.141 | 0.49 / 0.50 / 0.47 / 0.47 | 111 |
| ORACLE | 801 | 0.910 | 0.867 / 0.891 / 0.900 / 0.922 | 1.144 | 0.49 / 0.48 / 0.50 / 0.48 | 111 |
| ORACLE | 802 | 0.910 | 0.850 / 0.894 / 0.902 / 0.923 | 1.146 | 0.49 / 0.47 / 0.48 / 0.47 | 111 |
| ORACLE | 803 | 0.913 | 0.905 / 0.893 / 0.900 / 0.921 | 1.146 | 0.49 / 0.50 / 0.48 / 0.48 | 111 |
| SINGLE | 800 | 0.746 | 0.699 / 0.713 / 0.726 / 0.762 | 1.254 | 0.49 / 0.51 / 0.53 / 0.54 | 106 |
| SINGLE | 801 | 0.741 | 0.689 / 0.701 / 0.716 / 0.760 | 1.258 | 0.49 / 0.51 / 0.50 / 0.49 | 106 |
| SINGLE | 802 | 0.662 | 0.516 / 0.590 / 0.619 / 0.704 | 1.296 | 0.49 / 0.49 / 0.50 / 0.49 | 106 |
| SINGLE | 803 | 0.661 | 0.518 / 0.591 / 0.618 / 0.702 | 1.295 | 0.49 / 0.50 / 0.50 / 0.49 | 106 |

USABLE (ORACLE >= 0.9 and SINGLE <= ORACLE - 0.2 on >= 3/4 seeds): 2/4 seeds -> NOT USABLE
  ORACLE: SET median 0.912 (range 0.910-0.918); wall time per run median 111 min
  SINGLE: SET median 0.701 (range 0.661-0.746); wall time per run median 106 min
