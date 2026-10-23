# S90 phase 2 per run (EXPLORATORY, not a result)

Each firing: update (probe accuracy) c* -> c0, labelled ok or not, the key-position and all-position pairs and their labels, and the stream -> channel map before it.

| arm | seed | outcome | transition | firings |
|---|---|---|---|---|
| BASE | 3024 | MERGED (3 share) | None | 4800 (0.02) 11->0 ok; key 11->0 True, all 11->10 True; map [11, 11, 11, 11, 11, 11, 11, 11]<br>14400 (0.16) 0->9 ok; key 0->9 True, all 0->3 True; map [10, 0, 10, 1, 0, 0, 10, 0]<br>26400 (0.33) 9->8 ok; key 9->8 True, all 9->0 True; map [10, 9, 10, 1, 9, 11, 10, 9] |
| BASE | 3025 | BOUND ROUTED | 20400 | 19200 (0.81) 14->0 ok; key 14->0 True, all 14->0 True; map [14, 15, 10, 8, 12, 9, 14, 2] |
| BASE | 3026 | BOUND ROUTED | 15600 | 4800 (0.03) 11->15 ok; key 11->15 True, all 11->15 True; map [11, 11, 0, 11, 0, 11, 0, 0] |
| BASE | 3027 | MERGED (2 share) | None | 7200 (0.05) 14->1 ok; key 14->1 True, all 14->1 True; map [5, 14, 5, 6, 14, 14, 14, 5]<br>19200 (0.67) 1->12 ok; key 1->12 True, all 1->4 True; map [5, 1, 15, 6, 14, 11, 1, 5]<br>26400 (0.69) 5->4 ok; key 5->4 True, all 5->10 True; map [5, 1, 15, 6, 14, 11, 1, 5] |
| BASE | 3028 | MERGED (2 share) | None | 4800 (0.05) 10->5 ok; key 10->5 True, all 10->5 True; map [4, 10, 10, 10, 4, 10, 10, 10]<br>9600 (0.08) 2->14 ok; key 2->14 True, all 2->14 True; map [4, 10, 10, 2, 4, 2, 2, 2]<br>19200 (0.34) 14->7 ok; key 14->7 True, all 10->6 True; map [4, 10, 10, 14, 4, 2, 2, 14] |
| BASE | 3029 | BOUND ROUTED | 25200 | 4800 (0.06) 0->9 ok; key 0->9 True, all 0->9 True; map [14, 0, 0, 0, 4, 14, 0, 0]<br>12000 (0.22) 9->3 ok; key 9->3 True, all 9->3 True; map [4, 9, 4, 0, 4, 1, 9, 9]<br>24000 (0.83) 9->15 ok; key 9->15 True, all 9->6 True; map [7, 9, 11, 0, 4, 1, 3, 9] |
| BASE | 3030 | BOUND ROUTED | 30000 | 21600 (0.73) 2->14 ok; key 2->14 True, all 0->14 False; map [2, 15, 3, 2, 3, 11, 6, 13]<br>26400 (0.73) 3->12 ok; key 3->12 True, all 3->1 True; map [14, 15, 3, 14, 3, 11, 6, 13] |
| BASE | 3031 | MERGED (2 share) | None | 9600 (0.17) 1->5 ok; key 1->5 True, all 1->10 True; map [1, 9, 1, 7, 1, 9, 13, 7]<br>19200 (0.42) 7->10 ok; key 7->10 True, all 9->10 True; map [1, 9, 5, 7, 1, 9, 13, 7]<br>24000 (0.58) 9->6 ok; key 9->6 True, all 9->6 True; map [1, 9, 5, 7, 1, 9, 13, 10] |
| BASE | 3032 | BOUND ROUTED | 18000 | 4800 (0.03) 15->10 ok; key 15->10 True, all 1->10 True; map [15, 15, 1, 1, 1, 1, 15, 15]<br>9600 (0.08) 1->4 ok; key 1->4 True, all 1->4 True; map [14, 14, 1, 1, 12, 1, 14, 10]<br>16800 (0.81) 14->5 ok; key 14->5 True, all 14->13 True; map [14, 14, 1, 11, 12, 4, 15, 10] |
| BASE | 3033 | BOUND ROUTED | 19200 | 14400 (0.72) 4->15 ok; key 4->15 True, all 4->15 True; map [4, 11, 5, 10, 3, 10, 4, 6] |
| DELTA_SPLIT | 3024 | BOUND ROUTED | 34800 | 4800 (0.02) 10->15 NOT ok; key 10->15 False, all 13->5 True; map [13, 3, 12, 13, 12, 14, 8, 14]<br>9600 (0.06) 1->5 NOT ok; key 1->5 False, all 2->14 True; map [13, 2, 13, 13, 11, 2, 11, 1]<br>21600 (0.88) 7->0 NOT ok; key 7->0 False, all 7->0 False; map [15, 1, 12, 13, 8, 9, 3, 7] |
| DELTA_SPLIT | 3025 | BOUND ROUTED | 26400 | 21600 (0.94) 15->2 NOT ok; key 15->2 False, all 1->2 False; map [8, 3, 12, 13, 10, 15, 4, 14] |
| DELTA_SPLIT | 3026 | BOUND ROUTED | 24000 | 4800 (0.06) 2->6 ok; key 2->6 True, all 1->0 False; map [13, 2, 13, 2, 2, 2, 10, 13] |
| DELTA_SPLIT | 3027 | BOUND ROUTED | 40800 | 7200 (0.02) 5->1 ok; key 5->1 True, all 3->11 False; map [5, 6, 5, 10, 8, 5, 6, 2]<br>28800 (0.88) 1->3 NOT ok; key 1->3 False, all 11->3 False; map [5, 11, 10, 6, 4, 1, 0, 14]<br>40800 (0.92) 11->12 NOT ok; key 11->12 False, all 11->12 False; map [5, 11, 7, 6, 4, 1, 0, 14] |
| DELTA_SPLIT | 3028 | BOUND ROUTED | 30000 | 4800 (0.05) 1->5 ok; key 1->5 True, all 1->5 True; map [1, 14, 14, 2, 1, 2, 14, 14]<br>24000 (0.80) 0->3 NOT ok; key 0->3 False, all 8->3 False; map [0, 4, 14, 8, 13, 5, 2, 11]<br>28800 (0.84) 8->15 NOT ok; key 8->15 False, all 8->15 False; map [3, 4, 14, 8, 13, 1, 12, 11] |
| DELTA_SPLIT | 3029 | BOUND ROUTED | 24000 | 4800 (0.06) 5->10 NOT ok; key 5->10 False, all 0->14 True; map [0, 9, 0, 4, 4, 13, 0, 11]<br>9600 (0.09) 8->0 NOT ok; key 8->0 False, all 8->0 False; map [14, 7, 14, 14, 8, 14, 5, 14] |
| DELTA_SPLIT | 3030 | BOUND ROUTED | 26400 | 4800 (0.03) 15->1 ok; key 15->1 True, all 15->1 True; map [4, 15, 15, 15, 0, 0, 0, 15]<br>16800 (0.77) 2->4 NOT ok; key 2->4 False, all 2->4 False; map [2, 1, 9, 14, 13, 3, 10, 5]<br>26400 (0.94) 12->14 NOT ok; key 12->14 False, all 12->13 False; map [2, 15, 9, 12, 13, 8, 0, 11] |
| DELTA_SPLIT | 3031 | BOUND ROUTED | 40800 | 4800 (0.03) 7->9 ok; key 7->9 True, all 7->12 True; map [7, 7, 7, 7, 7, 7, 7, 7]<br>9600 (0.02) 4->8 NOT ok; key 4->8 False, all 9->8 True; map [9, 7, 10, 9, 9, 10, 9, 7]<br>19200 (0.23) 3->11 NOT ok; key 3->11 False, all 3->11 False; map [1, 14, 13, 8, 6, 0, 9, 3] |
| DELTA_SPLIT | 3032 | BOUND ROUTED | 33600 | 7200 (0.06) 10->14 ok; key 10->14 True, all 10->14 True; map [15, 10, 10, 1, 1, 15, 1, 10]<br>12000 (0.09) 7->4 ok; key 7->4 True, all 7->4 True; map [6, 7, 7, 7, 7, 6, 7, 1] |
| DELTA_SPLIT | 3033 | BOUND ROUTED | 28800 | 7200 (0.08) 5->14 NOT ok; key 5->14 False, all 3->10 False; map [14, 5, 14, 5, 15, 5, 15, 5] |
| DELTA_NOSPLIT | 3024 | BOUND ROUTED | 18000 | -- |
| DELTA_NOSPLIT | 3025 | BOUND ROUTED | 25200 | -- |
| DELTA_NOSPLIT | 3026 | BOUND ROUTED | 28800 | -- |
| DELTA_NOSPLIT | 3027 | STREAM-PARTIAL one-to-one | None | -- |
| DELTA_NOSPLIT | 3028 | STREAM-PARTIAL one-to-one | None | -- |
| DELTA_NOSPLIT | 3029 | STREAM-PARTIAL one-to-one | None | -- |
| DELTA_NOSPLIT | 3030 | BOUND ROUTED | 33600 | -- |
| DELTA_NOSPLIT | 3031 | BOUND ROUTED | 18000 | -- |
| DELTA_NOSPLIT | 3032 | BOUND ROUTED | 30000 | -- |
| DELTA_NOSPLIT | 3033 | BOUND ROUTED | 19200 | -- |
