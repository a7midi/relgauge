# Algebra audit patch

Adds `relgauge.algebraaudit`, a stage-1 analysis tool for saved blind-selection winners.

It loads a `--save-winners` pickle and, for each interface edge, restricts the target local rule table to the interface input while holding all other inputs fixed. Bijections `[q] -> [q]` are treated as permutations. The tool generates the finite permutation group produced by all such local permutations and classifies it as Cq/Zq-like, V4, D4, A4, S4, etc. This tells you whether the selected exact interface label behaves like a cyclic phase, a full permutation label, or something weaker.

Example:

```bash
python -m relgauge.algebraaudit example_results/blind_winners_q4.pkl ^
  --top-n 18 ^
  --out example_results/algebra_audit_q4.csv ^
  --plot example_results/fig_algebra_audit_q4.png
```

The main fields are:

- `group_name`
- `group_order`
- `group_zq_like`
- `group_abelian`
- `group_cyclic`
- `group_transitive`
- `permutation_fraction`
- `edge_summaries`

Interpretation:

- `group_zq_like=True` supports a finite Z_q phase interpretation.
- `group_name=S4` means full label permutation symmetry, not U(1)-like electromagnetism.
- low `permutation_fraction` means exact S exists, but the raw local rule tables do not act permutively on the interface alphabet.
