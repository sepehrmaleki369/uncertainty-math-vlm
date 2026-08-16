# real_parameter_sensitivity_full.csv

Regenerated locally from `real_parameter_sensitivity_full_item_scores.csv`
(private) with `pilot.parameter_sensitivity.summarize_grid`, the same function
the notebook used, so this table is reproducible from the item-level scores
rather than only copied out of a session.

**One deliberate difference from the file the Colab run wrote to Drive.** The
three K=10 rows differ in the fourth decimal (6.7e-05, 2.2e-04, 6.2e-04); K=3
and K=5 are identical to the last digit. The entropies agree to twelve decimal
places -- `n_entropy_values` matches exactly at 34/35/31 -- so this is
float-level tie-breaking, not a different result. AUROC scores a tie as half
credit, K=10 produces many near-identical entropies, and a last-bit difference
from the float round trip flips such a pair from tied to ordered.

The paper quotes no K=10 figure, so nothing reported depends on which version is
used. The reproducible one is kept here on principle: an artifact a reader
cannot regenerate from the data shipped beside it is worth less than one they
can.

The claim-bearing snapshot is `reference/parameter_sensitivity_20260816.json`,
which carries the artifact controls as well and is asserted by
`pilot/tests/test_parameter_sensitivity_run.py`.
