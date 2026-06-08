# finiteobserver patch for relgauge

Place files as:

```text
relgauge/finiteobserver.py
tests/test_finiteobserver.py
```

Optionally expose the module from `relgauge/__init__.py`:

```python
from . import core, quotients, observables, experiments, stats, report, finiteobserver
__all__ = ["core", "quotients", "observables", "experiments", "stats", "report", "finiteobserver"]
```

Validate:

```bash
python -m pytest tests -q
python -m relgauge.finiteobserver 4 --ks-B 2,3,4 --ks-A 2,3 --ws 1 --horizons 1,2,3 --instances 20 --out example_results/finite_observer.csv
```

The old `relgauge.arealaw` active resolver remains useful as an infinite-memory upper bound. This module replaces it for the closure-respecting physical observer test: the observer is an actual downstream SCC with exactly `q**kA` internal memory states, and only its final internal state after `T` ticks is read.
