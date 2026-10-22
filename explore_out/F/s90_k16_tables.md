### The type-level perfect gate at k = 16 (explore_f_s90_k16; seed 3060; SLOW)

| task | memory | budget | k = 16 type-level gate (3060) | end margin / separated | per-sequence gate, k = 2 (2/2) |
|---|---|---|---|---|---|
| CTXPOOL | HEBBCONV | 24000 | bound @2400 (final 1.000) | 1.00 / 1.00 | O_HEBBCONV_CTXPOOL: 3060 @2400; 3061 @4800 |
| CTXPOOL | DELTA | 24000 | bound @1200 (final 1.000) | 1.00 / 1.00 | O_DELTA_CTXPOOL: 3060 @1200; 3061 @2400 |
| ALLPOOL | HEBBCONV | 48000 | bound @7200 (final 0.985) | 1.00 / 1.00 | O_HEBB_CONV_ALLPOOL_48K: 2800 @6000; 2801 @28800 |
| ALLPOOL | DELTA | 24000 | bound @3600 (final 0.988) | 1.00 / 1.00 | O_DELTA_ALLPOOL: 2800 @3600; 2801 @3600 |

Curves (held-out accuracy every 2400):

- TL16_HEBBCONV_CTXPOOL|3060: 1.00 1.00
- TL16_DELTA_CTXPOOL|3060: 1.00
- TL16_HEBBCONV_ALLPOOL|3060: 0.32 0.88 0.96 0.98
- TL16_DELTA_ALLPOOL|3060: 0.92 0.99

READING (fixed before the runs): **the type-level gate binds at k = 16**

F/s90_k16: 4 runs ok (failed records 0), 0.28 h of run time; CPUs ['Intel(R) Xeon(R) Processor @ 2.80GHz']; code SHA 81f8f8098645
