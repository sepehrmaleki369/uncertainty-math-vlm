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

## The two figures, and why there are two

`real_parameter_sensitivity_full.png` is the figure the notebook drew. It plots
**uncontrolled** AUROC against temperature and it rises, peaking at 0.892 at
T=1.0, K=5. Nothing in it is wrong, but read alone it argues for hotter
sampling, which is the opposite of the finding it belongs to.

`real_parameter_sensitivity_controls.png` puts that panel beside the same grid
after removing parse failures and ceiling-entropy items, on one shared y-axis.
The rise flattens: at K=5 the controlled figures are 0.741, 0.749 and 0.722, and
the intervals overlap. The correction tracks the artifacts rather than the
temperature, since the parse-failure rate runs 0.5% -> 4.8% -> 12.7% and the
controls delete two thirds of the items at T=1.0.

Both are kept. The first is what the run produced and is the honest record of
it; the second is what a reader should take away, and the paper reports no
temperature effect. Regenerate the second with
`python reference/wacv_evaluation_artifact/build_sensitivity_controls_figure.py`,
which reads only the committed snapshot.
