### S89-F follow-up pilots (explore_f_s89_post; seeds 2800-2801; Hebbian, k = 2, SLOW, 24000)

Per run: transition or final accuracy; end margin; η² by stream at KEY pooled / within-sequence; VAL cos pooled / within-sequence; fraction of probe sequences with η²_seq > 0.9; accuracy when the queried pair was / was not its key's last write.

| arm | seed | outcome | margin | η² KEY (pooled / within) | VAL cos (pooled / within) | sequences routed | last / not last |
|---|---|---|---|---|---|---|---|
| O_TYPE_HEBB_ALLPOOL | 2800 | unbound 0.278 | 0.51 | 0.00 / 0.51 | 1.00 / 0.49 | 0.51 | 0.32 / 0.27 |
| O_TYPE_HEBB_ALLPOOL | 2801 | unbound 0.275 | 0.49 | 0.01 / 0.49 | 0.99 / 0.51 | 0.49 | 0.30 / 0.27 |
| O_HEBB_CONV_POOL | 2800 | bound @4800 | 1.00 | 1.00 / 1.00 | 0.00 / 0.00 | 1.00 | 0.99 / 0.98 |
| O_HEBB_CONV_POOL | 2801 | bound @13200 | 1.00 | 1.00 / 1.00 | 0.00 / 0.00 | 1.00 | 0.96 / 0.96 |
| O_HEBB_CONV_ALLPOOL | 2800 | bound @6000 | 1.00 | 1.00 / 1.00 | 0.00 / 0.00 | 1.00 | 0.99 / 0.99 |
| O_HEBB_CONV_ALLPOOL | 2801 | unbound 0.872 | 1.00 | 1.00 / 1.00 | 0.00 / 0.00 | 1.00 | 0.86 / 0.87 |
| O_HEBB_CONV_ALLPOOL_48K | 2800 | bound @6000 | 1.00 | 1.00 / 1.00 | 0.00 / 0.00 | 1.00 | 0.99 / 0.99 |
| O_HEBB_CONV_ALLPOOL_48K | 2801 | bound @28800 | 1.00 | 1.00 / 1.00 | 0.00 / 0.00 | 1.00 | 0.95 / 0.97 |

Curves (held-out accuracy every 2400):

- O_TYPE_HEBB_ALLPOOL|2800: 0.27 0.26 0.26 0.27 0.27 0.27 0.28 0.28 0.25 0.28
- O_TYPE_HEBB_ALLPOOL|2801: 0.26 0.27 0.28 0.26 0.26 0.27 0.26 0.27 0.27 0.27
- O_HEBB_CONV_POOL|2800: 0.32 0.96 0.98
- O_HEBB_CONV_POOL|2801: 0.32 0.71 0.78 0.89 0.94 0.96
- O_HEBB_CONV_ALLPOOL|2800: 0.32 0.93 0.99
- O_HEBB_CONV_ALLPOOL|2801: 0.31 0.67 0.73 0.76 0.77 0.78 0.78 0.77 0.85 0.87
- O_HEBB_CONV_ALLPOOL_48K|2800: 0.32 0.93 0.99
- O_HEBB_CONV_ALLPOOL_48K|2801: 0.31 0.67 0.73 0.76 0.77 0.78 0.78 0.77 0.85 0.87 0.92 0.95 0.97

Conv pilots: 3/4 [0.30, 0.95] MAJORITY bound (O_HEBB_CONV_POOL 2/2, O_HEBB_CONV_ALLPOOL 1/2).

READING (fixed before the runs): **partly**

Fallback on ALLPOOL (O_HEBB_CONV_ALLPOOL at 48000; the conv configuration is the round's valid Hebbian configuration on POOL): 2/2 [0.34, 1.00] RELIABLE bound.

READING (fixed before the 2 runs): **ALLPOOL: the conv configuration needs the longer budget**

F/s89_post: 8 runs ok (failed records 0), 0.76 h of run time; CPU ['Intel(R) Xeon(R) Processor @ 2.10GHz', 'Intel(R) Xeon(R) Processor @ 2.80GHz']; code SHA 267d08456c38
