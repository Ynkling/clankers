### S69 forget gate on a single Hebbian channel — counts (BOUND; Wilson 95%; band)

| arm | n | BOUND | transitions: median (all) | final acc median | write order acc last / not-last | S59 delta, same seeds |
|---|---|---|---|---|---|---|
| S69_FIXED | 10/10 | 0/10 [0.00, 0.28] NEVER | -- | 0.22 | 0.21 / 0.23 | 3/10 |
| S69_ROUTED | 10/10 | 0/10 [0.00, 0.28] NEVER | -- | 0.17 | 0.17 / 0.16 | 9/10 |
| S69_GDN | 10/10 | 0/10 [0.00, 0.28] NEVER | -- | 0.17 | 0.17 / 0.16 | 8/10 |
| S69_ORACLE | 2/2 | 1/2 [0.09, 0.91] MAJORITY | 31200 (31200) | 0.88 | 0.88 / 0.88 | --/10 |

Validity: S69_ORACLE 1/2 (transitions [None, 31200]) -> VALID

READING S69: **the forget gate needs the delta rule** (Hebbian FIXED 0/10, ROUTED 0/10, GDN 0/10; rules: needs delta if ROUTED and GDN <= FIXED + 1; alone suffices if either >= 8/10)

Paired with S59's delta single channel on the same seeds and batches (BOUND; delta only / hebb only): FIXED: S59 delta 3/10 vs Hebbian 0/10; ROUTED: S59 delta 9/10 vs Hebbian 0/10; GDN: S59 delta 8/10 vs Hebbian 0/10

Learned decay at the end (median α per layer at CTX/KEY/VAL):

- S69_FIXED: L1 ch0 0.950/0.950/0.950; L2 ch0 0.950/0.950/0.950; L3 ch0 0.950/0.950/0.950
- S69_ROUTED: L1 ch0 0.975/0.999/0.982; L2 ch0 0.944/0.997/0.966; L3 ch0 0.886/0.990/0.914
- S69_GDN: L1 ch0 0.988/0.999/0.990; L2 ch0 0.943/0.997/0.952; L3 ch0 0.684/0.994/0.926
- S69_ORACLE: L1 other 0.950/0.950/0.950, own 0.950/0.950/0.950; L2 other 0.950/0.950/0.950, own 0.950/0.950/0.950; L3 other 0.950/0.950/0.950, own 0.950/0.950/0.950

Runs: 32 ok (failed records 0), 11.28 h of run time; CPU ['Intel(R) Xeon(R) Processor @ 2.10GHz']; code SHA ff908bdbc0b2
