# Errata found in the OSTIR monograph v1.0

Confirmed by `tests/test_theory.py` and `results/*.json`. Each is reproducible
from the closed-form layer alone.

1. **Appendix B, two entries.** Exact numerical integration of `E[max]` of n
   standard normals matches 9 of 11 tabulated values to <0.001, but
   `n=64` (table 2.320, exact 2.3437) and `n=128` (table 2.580, exact 2.5946)
   are wrong. These feed Table 2.1 at G=64 and G=128. Correct
   `D(4,64) = 0.008138`, not 0.007982 — the deck's Path B operating point is
   ~2% more lossy than stated (-0.08 dB).

2. **Cor 4.4 arithmetic.** With `S_w = 1/0.085 = 11.7647` and the monograph's
   own `f` values (0.807 MHA, 0.944 GQA-8), composition gives **3.83x** and
   **7.33x**, not the printed 3.86x and 7.15x. The printed numbers imply
   `f = 0.8098` and `f = 0.9400`. The traffic model itself is fine — E7
   reproduces `f` to 4 decimals. Headline should be **3.8-7.3x**.

3. **Thm 2.9 is false as stated.** Per-group Lloyd-Max beats grouped min-max
   on clean Gaussian draws (where (A2) holds) by 2.9 dB at G=32, 1.8 dB at
   G=64. Necessarily so: Lloyd-Max is MSE-optimal for a given distribution
   and level count, so per-group Lloyd-Max is both adaptive and optimally
   shaped. The proof in §2.6 compares grouping against a **global** companded
   codebook — a different object — so it does not establish the claim.
   The conclusion holds in **rate-aware** form: a per-group codebook costs
   `2^b * 16 / G` bits/weight (12.0 bpw at G=32 vs min-max's 5.0), so
   min-max wins decisively at equal rate. §2.6 should assert that version.
   The practical advice (spend effort on outliers, not codebooks) survives.

Not errata, but worth stating: `r = 0.085` and every magnitude derived from
it are Sapphire-Rapids-specific. See [[platform-transferability]].
