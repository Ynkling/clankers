S75 at S = 4 (EXPLORATORY): ICMC S=4 V=4 NB=16 block words 6..14 M=176 words/source L=768; 24000 updates; split threshold 0.88; code SHA 5fb120a92d69; GPUs ['NVIDIA GeForce GTX 1660 SUPER']


VALIDITY: ORACLE final SET [0.892, 0.88] -> UNTESTED

| arm | ROUTED (Wilson; band) | median GAP | GAP by bin 1 / 2 / 3-4 / 5+ | median SET | splits (on target) | median eta (layers) | margin |
|---|---|---|---|---|---|---|---|
| ORACLE | 2/2 [0.34, 1.00] RELIABLE | 1.00 | 1.00 / 1.00 / 1.00 / 1.00 | 0.886 | - | 1.00 | 1.00 |
| SINGLE | 0/10 [0.00, 0.28] NEVER | 0.00 | 0.00 / 0.00 / 0.00 / 0.00 | 0.538 | - | - | - |
| WIDE_SPLIT | 0/10 [0.00, 0.28] NEVER | -0.01 | -0.03 / -0.01 / -0.01 / -0.01 | 0.534 | 30 (30) | 0.00 | 0.03 |
| REG3 | 0/10 [0.00, 0.28] NEVER | 0.00 | -0.00 / 0.00 / 0.00 / 0.00 | 0.539 | - | 0.00 / 0.00 / 0.00 | 0.02 |

READING (pre-fixed): UNTESTED  (ROUTED {'WIDE_SPLIT': 0, 'REG3': 0}; the better arm needs >= 8/10)
Printed: S74's usability count at S = 4: ORACLE mean 0.886 (>= 0.9: False); SINGLE <= ORACLE - 0.2 on 10/10 seeds

DIAGNOSTICS (medians at the end): eta^2 by source at word index 1-2 / 5+ (layer 1), probe L1 / L2 / L3 / final, REG3 beta^R at SRC / word, train minutes, peak GB
  ORACLE      eta 1-2 1.00 / 5+ 1.00; probe 0.25 / 0.25 / 0.25 / 0.25; train 26 min per run; peak 0.69 GB per batch of R = [1]
  SINGLE      eta 1-2 - / 5+ -; probe 0.25 / 0.25 / 0.26 / 0.26; train 25 min per run; peak 0.55 GB per batch of R = [1]
  WIDE_SPLIT  eta 1-2 0.00 / 5+ 0.00; probe 0.25 / 0.25 / 0.25 / 0.25; train 30 min per run; peak 0.98 GB per batch of R = [1]
  REG3        eta 1-2 0.00 / 5+ 0.00; probe 0.25 / 0.25 / 0.25 / 0.25; beta 0.80 / 0.15; train 38 min per run; peak 1.37 GB per batch of R = [1]

PER RUN: arm seed ROUTED SET GAP maps / eta / margin per layer, splits
  ORACLE      830 R 0.892 1.02 L1:[0, 1, 2, 3]/1.00/1.00
  ORACLE      831 R 0.880 0.98 L1:[0, 1, 2, 3]/1.00/1.00
  SINGLE      830 . 0.538 0.00 
  SINGLE      831 . 0.535 0.00 
  SINGLE      832 . 0.537 0.00 
  SINGLE      833 . 0.538 0.00 
  SINGLE      834 . 0.535 0.00 
  SINGLE      835 . 0.540 0.00 
  SINGLE      836 . 0.538 0.00 
  SINGLE      837 . 0.534 0.00 
  SINGLE      838 . 0.595 0.00 
  SINGLE      839 . 0.538 0.00 
  WIDE_SPLIT  830 . 0.535 -0.01 L1:[2, 2, 2, 2]/0.00/0.02 splits 3 at [9600, 14400, 19200]
  WIDE_SPLIT  831 . 0.533 -0.01 L1:[2, 0, 2, 2]/0.00/0.03 splits 3 at [7200, 12000, 16800]
  WIDE_SPLIT  832 . 0.535 -0.01 L1:[1, 1, 1, 1]/0.00/0.03 splits 3 at [7200, 12000, 16800]
  WIDE_SPLIT  833 . 0.534 -0.01 L1:[3, 3, 3, 3]/0.00/0.02 splits 3 at [7200, 12000, 16800]
  WIDE_SPLIT  834 . 0.533 -0.00 L1:[1, 1, 1, 0]/0.00/0.03 splits 3 at [7200, 12000, 16800]
  WIDE_SPLIT  835 . 0.533 -0.02 L1:[0, 0, 0, 0]/0.00/0.03 splits 3 at [7200, 12000, 16800]
  WIDE_SPLIT  836 . 0.536 -0.01 L1:[1, 1, 1, 1]/0.00/0.02 splits 3 at [7200, 12000, 16800]
  WIDE_SPLIT  837 . 0.529 -0.01 L1:[1, 3, 3, 3]/0.00/0.03 splits 3 at [7200, 12000, 16800]
  WIDE_SPLIT  838 . 0.536 -0.21 L1:[3, 1, 3, 3]/0.00/0.03 splits 3 at [7200, 12000, 16800]
  WIDE_SPLIT  839 . 0.534 -0.01 L1:[1, 1, 1, 1]/0.00/0.03 splits 3 at [7200, 12000, 16800]
  REG3        830 . 0.535 -0.01 L1:[2, 2, 2, 2]/0.00/0.02 L2:[2, 2, 2, 2]/0.00/0.02 L3:[2, 2, 2, 2]/0.00/0.03
  REG3        831 . 0.538 0.01 L1:[0, 0, 0, 2]/0.00/0.03 L2:[1, 1, 1, 1]/0.00/0.03 L3:[1, 1, 1, 1]/0.00/0.01
  REG3        832 . 0.538 0.00 L1:[3, 3, 3, 1]/0.00/0.03 L2:[2, 2, 2, 2]/0.00/0.03 L3:[2, 2, 2, 2]/0.00/0.00
  REG3        833 . 0.538 0.00 L1:[2, 3, 3, 2]/0.00/0.03 L2:[0, 0, 0, 0]/0.00/0.03 L3:[0, 0, 0, 0]/0.00/-0.00
  REG3        834 . 0.539 0.01 L1:[2, 1, 1, 2]/0.00/0.03 L2:[0, 0, 0, 3]/0.00/0.03 L3:[0, 0, 0, 0]/0.00/0.01
  REG3        835 . 0.539 -0.00 L1:[2, 1, 1, 2]/0.00/0.03 L2:[0, 3, 3, 0]/0.00/0.03 L3:[0, 0, 0, 0]/0.00/0.02
  REG3        836 . 0.539 0.00 L1:[1, 0, 1, 0]/0.00/0.03 L2:[3, 3, 3, 3]/0.00/0.03 L3:[3, 3, 3, 3]/0.00/0.02
  REG3        837 . 0.540 0.02 L1:[2, 2, 2, 2]/0.00/0.02 L2:[3, 3, 3, 3]/0.00/0.02 L3:[3, 3, 3, 3]/0.00/0.01
  REG3        838 . 0.539 -0.19 L1:[1, 1, 1, 1]/0.00/0.03 L2:[3, 3, 3, 3]/0.00/0.03 L3:[3, 3, 3, 3]/0.00/0.03
  REG3        839 . 0.588 0.14 L1:[2, 2, 2, 2]/0.00/0.03 L2:[3, 3, 3, 3]/0.00/0.02 L3:[3, 3, 3, 3]/0.00/0.01
