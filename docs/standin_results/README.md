# STAND-IN results - NOT PAN results

These files come from a smoke test of the code on `scripts/make_pseudo_mawsa.py` data (sentences cut out of
public-domain books, labelled as if they were authors; validation = 6 books never seen in training). The PAN
data could not be downloaded where this was prepared. Quick settings were used (20,000 training pairs,
6,727 evaluation pairs, 1 seed, short training). Do not quote any number from here in the paper.
Re-run `scripts/ablation.sh all` and `scripts/ablation.sh retrieval` on the real data.
