### S68 premise — counts (BOUND; Wilson 95%; band)

| arm | n | BOUND | transitions: median (all) | final acc median | write order acc last / not-last |
|---|---|---|---|---|---|
| S68_SINGLE_S4 | 10/10 | 2/10 [0.06, 0.51] MINORITY | 25800 (18000, 33600) | 0.41 | 0.42 / 0.41 |
| S68_ORACLE_S4 | 2/2 | 2/2 [0.34, 1.00] RELIABLE | 1200 (1200, 1200) | 1.00 | 1.00 / 1.00 |
| S68_SINGLE_S8 | 10/10 | 0/10 [0.00, 0.28] NEVER | -- | 0.42 | 0.47 / 0.42 |
| S68_ORACLE_S8 | 2/2 | 2/2 [0.34, 1.00] RELIABLE | 1200 (1200, 1200) | 1.00 | 1.00 / 1.00 |

Validity S=4: S68_ORACLE_S4 2/2 (transitions [1200, 1200]) -> VALID

Validity S=8: S68_ORACLE_S8 2/2 (transitions [1200, 1200]) -> VALID

READING S68: **neither** (SINGLE_DELTA S=4 2/10, S=8 0/10; rules: scale >= 7/10 at S=8; number of streams <= 3/10 at S=8 with >= 7/10 at S=4)
- S=4: one delta channel BOUND 2/10 [0.06, 0.51] MINORITY vs recorded gated Hebbian S54 B_HEBB BOUND ROUTED 19/20 [0.76, 0.99] RELIABLE (BOUND 20/20); counts only, other seeds; Fisher one-sided p(gated > single) = 6.36e-05
- S=8: one delta channel BOUND 0/10 [0.00, 0.28] NEVER vs recorded gated Hebbian S54 C_HEBB BOUND ROUTED 9/10 [0.60, 0.98] RELIABLE (BOUND 9/10); counts only, other seeds; Fisher one-sided p(gated > single) = 5.95e-05

Per seed (B@transition or final acc): S68_SINGLE_S4 300 0.26, 301 B@33600, 302 0.23, 303 0.25, 304 0.22, 305 0.55, 306 B@18000, 307 0.26, 308 0.74, 309 0.75; S68_SINGLE_S8 300 0.32, 301 0.30, 302 0.59, 303 0.29, 304 0.39, 305 0.46, 306 0.54, 307 0.56, 308 0.29, 309 0.53

Runs: 24 ok (failed records 0), 18.63 h of run time; CPU ['Intel(R) Xeon(R) Processor @ 2.10GHz']; code SHA ff908bdbc0b2
