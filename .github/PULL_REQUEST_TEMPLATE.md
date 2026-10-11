## What this changes

<!-- one paragraph: the test, screen, amendment or document; which branch it belongs on -->

## Pre-registration (main-line tests and screens)

- [ ] The specification (`specs/test_<name>.md`) or the docstring's readings were committed **before the first run**; commit: `…`
- [ ] First record's time (from the results file's `meta.started`): `…`
- [ ] Seeds: `…` (checked against every branch; no collision)
- [ ] Outcomes are fixed by a named function; each reading rests on exactly one outcome
- [ ] Every amendment is dated and states what had been seen when it was made

## Records

- [ ] Results file under `results/<machine>/` with its row in `results/README.md`
- [ ] All CHECKs passed (or: which were printed rather than asserted, and why)
- [ ] The run's code was not changed while it ran

## Text

- [ ] Every number in the text comes from a record a script can recompute
- [ ] Readings are quoted as the docstring prints them; descriptions and interpretations are labelled as such
- [ ] Exploratory outputs carry "EXPLORATORY, not a result"
