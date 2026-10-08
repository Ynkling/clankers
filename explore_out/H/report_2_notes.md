**Which readings apply.** Both screens are VALID: the oracles bound 2/2 (S58b at updates 3600 and 2400; S60's
ceiling_conv at 1200).
- **S58b "does not carry".** At eight streams the split bound routed 10/10. The two best S58 arms did not come close:
  RESET 2/10 (8 vs 0 discordant, one-sided p = 0.0039), GUMBEL_W 0/10 (10 vs 0, p = 0.00098).
- **S60: no device changed discovery at n = 10.**
- **D3 (does EMA evaluation shrink the BOUND asymmetry?) is uninformative.** Neither REF nor EMA_EVAL had a run that bound on a single final evaluation, or a run whose accuracy flipped back below 0.95.

**Unexpected / worth a look**
- **At eight streams the row copy is the active part of the split, not the reset.** RESET fired under the same plateau rule
  as SPLIT: 3 operations in every run, every target labelled correct (c* held >= 2 streams, c0 none). It still ended
  MERGED (2 share) on 8/10 seeds. SPLIT bound routed on all 10, by 27600. At four streams (S58) the two arms could not be
  told apart (the split itself did nothing there). The audit 2.3 question is answered for eight streams only: zeroing W_g's
  Adam state alone does not reproduce the split.
- **Write-side Gumbel noise makes eight-stream merges worse.** All 10 GUMBEL_W runs merged, with more sharing than the
  other arms (end maps up to 3+2+2+1, 4+3+1 and 6+2), against 2+1+...+1 for RESET. Noise helped nowhere in this session.
- **The saturation criterion (max prob > 0.99 at > 90% of positions) never fires in a run that binds.** Discovered runs sit
  at a constant 0.65 of probe positions above 0.99 from the transition on, in every arm. The failed runs (KEY,
  STREAM-PARTIAL) are the ones that reach 0.9-1.0. So in these runs saturation marks a committed wrong partition. It is not
  a plateau before binding, and Prieto et al.'s mechanism has no handle here. STABLEMAX starts much softer (0.01-0.38 at
  1200 against REF's 0.52-0.65) but binds at the same updates.
- **TEMP_FLOOR's learned temperature ran straight to the floor.** In every run τ fell to 0.47-0.50 by the first evaluation
  and stayed there. The clamp max(τ, 0.5) cuts the gradient below 0.5, so this arm is in effect "logits x 2". The gate
  wants to be sharper, not softer.
- **EMA_EVAL had the same outcome as REF on every seed (8/10, same seeds).** Its transitions are 1200 later on the 6 seeds
  where REF bound at 2400 (the EMA lags about 1000 updates); training is REF's bit for bit by construction.
- **ORACLE_PLAIN** (the perfect gate without the conv layer) again did not bind this layout (final accuracy 0.32 and 0.53),
  as batch 18 found. Validity rests on ceiling_conv, as fixed in the docstring.

**Overall (exploratory).** Nothing from this session replaces the KEYMASS split. At four streams, k = 16, every arm except
GUMBEL_RW is at the ceiling. At eight streams the split works on its own and its continuous stand-ins do not. What
matters is the row copy (S24's mechanism: the copy makes the busy and idle channels' logits nearly tie). Candidate for the
main line: a pre-registered SPLIT vs RESET at eight streams on disjoint seeds, to confirm "the copy, not the reset".
