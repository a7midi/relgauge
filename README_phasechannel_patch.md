# phasechannel.py patch

Place:

```text
relgauge/phasechannel.py
tests/test_phasechannel.py
```

Requires the earlier `relgauge/finiteobserver.py` patch.

Run:

```bash
python -m pytest tests -q
python -m relgauge.phasechannel 4 --ks-B 2,3,4 --ks-A 3 --ws 1,2,3 --horizons 1,3,6 --instances 20 --out example_results/phase_channel.csv --plot example_results/fig_phase_channel.png
```

The default tests two phase variables:

- `cut`: first-updated B vertex, i.e. schedule-cut phase.
- `erased_value`: the q-ary old value of phase vertex 0 when that vertex is updated first.

Optional:

```bash
python -m relgauge.phasechannel 4 --phase-modes cut,erased_value,cut_value --protocols first,same
```

`first` means the phase is imposed only on the first tick and later schedules are gauge noise.
`same` repeats the same schedule cut at every tick and is a phase-retention upper-bound/control.
