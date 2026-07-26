# Cache-Resident Quantized Inference on General-Purpose CPUs

### A Formal Treatment of the OSTIR Thesis: Rate Algebra, Rate–Distortion Bounds, Residency Laws, and an Executable Validation Protocol

**Version 1.0 — July 2026**
**OSTIR Labs, Inc. — Internal Technical Monograph**

---

## Abstract

The OSTIR thesis asserts that quantizing model weights to a sufficiently low effective bit rate makes the per-core working set resident in L2 cache, converting a DRAM-bandwidth-bound decode loop into an L2-bandwidth-bound one, with an approximately $12\times$ throughput consequence. This document makes that thesis precise, proves what can be proven, falsifies what is false, and reduces the remainder to six experiments executable on a single socket.

Four results dominate:

1. **The deck's arithmetic is exactly right, and its implied slice is exactly recoverable.** All four points on the slice-size scatter lie on a single line with $N_{\text{slice}} = 3.614 \times 10^6$ weights (§2.4). The bit-rate algebra is correct to the last digit.
2. **"The 11%" has a closed-form meaning, and it is nearly exhausted.** The metadata fraction $\varphi = 1/9 = 11.11\%$ is simultaneously the marginal elasticity of cache capacity with respect to group size and the capacity gained by the last group-size doubling (Prop. 2.6, Thm. 2.7). Total remaining headroom from *all* further metadata compression is $12.5\%$ of capacity — and realistically about a third of that. **Metadata compression is a dead end. Re-tiling is not** (§3.6).
3. **The residency law is brutally nonlinear.** Speedup obeys $S(h) = [1 - h(1-r)]^{-1}$. Achieving $10\times$ requires a hit rate $h \geq 0.984$. At $h = 0.95$ you get $7.6\times$; at $h = 0.90$, $5.7\times$. One percentage point of hit rate near the operating point is worth half a turn of speedup (Thm. 4.2). **The engineering problem is not compressing 11%; it is eliminating the last 2% of misses.**
4. **End-to-end decode speedup is 4–7×, not 12×** — because weights are only $f \approx 0.81$–$0.94$ of decode traffic and the KV cache cannot be co-resident (Thm. 3.4, Cor. 4.4). This is still an excellent number. It should be the number in the deck.

A fifth result reverses a common assumption: **on CPU, KV recompute is never competitive with KV fetch**, by a margin of roughly $350\times$ (Thm. 5.3). The crossover bandwidth is $\approx 8$ MB/s. This is the opposite of the GPU regime and it simplifies the runtime policy engine dramatically.

---

## Table of Contents

- **Part 0** — Notation, assumptions, and standing conventions
- **Part I** — The bit-rate algebra: effective rate, hierarchical metadata, and the recovery of $N_{\text{slice}}$
- **Part II** — Rate–distortion theory of grouped quantization: extreme-value ranges, Bennett's integral, Panter–Dite, Lloyd–Max, and the outlier result
- **Part III** — The residency constraint: working-set decomposition, the KV impossibility theorem, and the blocked-GEMM reformulation
- **Part IV** — The performance model: roofline, AMX utilization, the residency law, and Amdahl composition
- **Part V** — Algorithms: the group-size solver, the hierarchical quantizer, outlier extraction, LUT dequantization, and the KV policy engine
- **Part VI** — Validation protocol: six experiments, exact counters, pass/fail thresholds
- **Part VII** — What is actually novel, and why
- **Appendices** — Symbol table, extreme-value table, derivation notes, references

---

# Part 0 — Notation and Standing Assumptions

## 0.1 Symbols

| Symbol | Meaning | Units |
|---|---|---|
| $b$ | Code width per weight | bits |
| $G$ | Quantization group size | weights |
| $m$ | Metadata bits per group | bits |
| $B_{\text{eff}}$ | Effective bit rate per weight | bits/weight |
| $\varphi$ | Metadata fraction of the bit budget | — |
| $N$, $N_{\text{slice}}$ | Weights in the per-core resident panel | weights |
| $C$ | L2 capacity per core | bytes |
| $\eta$ | Usable fraction of L2 | — |
| $m_c, k_c$ | GEMM panel dimensions ($N = m_c k_c$) | — |
| $L$ | Context length | tokens |
| $n_\ell$ | Layers | — |
| $h_{kv}$ | KV heads (post-GQA) | — |
| $d_h$ | Head dimension | — |
| $\beta_{L2}, \beta_{DR}$ | Per-core L2 / DRAM bandwidth | B/cycle |
| $r$ | Bandwidth ratio $\beta_{DR}/\beta_{L2}$ | — |
| $h$ | L2 hit rate on the weight stream | — |
| $S$ | Speedup | — |
| $f$ | Weight share of per-token traffic | — |
| $D$ | Mean squared quantization error | $\sigma^2$ units |
| $\sigma^2$ | Per-group weight variance | — |
| $I$ | Arithmetic intensity | ops/byte |
| $\pi$ | Compute peak per core | ops/cycle |

## 0.2 Standing assumptions

**(A1) High-rate regime.** All distortion analysis assumes $b \geq 3$, where the high-resolution approximation $D \approx \Delta^2/12$ holds to within a few percent. At $b = 2$ the approximation degrades and results are indicative only.

**(A2) Locally Gaussian weights.** Within a group of $G$ weights drawn from one row/column block, weights are approximately i.i.d. $\mathcal{N}(0, \sigma^2)$. This is well supported empirically for transformer weight matrices *excluding outlier channels*, which are handled separately in §2.7. It is a worse assumption for KV, which is heavier-tailed.

**(A3) Symmetric distribution.** $\mathbb{E}[\text{range of } G] = 2\,\mathbb{E}[\max_G]$.

**(A4) Bandwidth-bound decode.** At batch size 1, the decode loop is memory-bound; §4.3 proves this and derives the batch size at which it ceases to hold.

**(A5) Steady-state, single-socket, NUMA-local.** All bandwidth figures are per-core shares in steady state with the working set NUMA-local and turbo/C-states pinned.

---

# Part I — The Bit-Rate Algebra

## 1.1 Affine grouped quantization

Partition a weight tensor into contiguous groups of $G$ weights. For group $g$ with values $\{w_i\}_{i=1}^G$, define

$$
s_g = \frac{\max_i w_i - \min_i w_i}{2^b - 1}, \qquad z_g = \min_i w_i
$$

and encode

$$
q_i = \operatorname{clamp}\!\left(\left\lfloor \frac{w_i - z_g}{s_g} \right\rceil,\, 0,\, 2^b - 1\right), \qquad \hat{w}_i = s_g q_i + z_g .
$$

Storage per group is $bG$ code bits plus $m$ metadata bits. With $s_g, z_g$ each in fp16, $m = 32$.

**Definition 1.1 (Effective bit rate).**
$$
\boxed{\;B_{\text{eff}}(b, G) = b + \frac{m}{G}\;}
$$

**Definition 1.2 (Metadata fraction).**
$$
\varphi(b,G) = \frac{m/G}{B_{\text{eff}}(b,G)} = \frac{m}{bG + m}
$$

## 1.2 The deck's operating points, verified

With $b = 4$, $m = 32$:

| $G$ | $B_{\text{eff}}$ | $\varphi$ |
|---|---|---|
| 16 | 6.000 | 33.3% |
| 32 | 5.000 | 20.0% |
| 64 | **4.500** | **11.11%** |
| 128 | 4.250 | 5.88% |
| 256 | 4.125 | 3.03% |

Slide 7's Path A ($5.0$) and Path B ($4.5$) are exact. The stated identity "metadata amortized $2\times$: $1.0 \to 0.5$ bit/w" is exact: $m/32 = 1.0$, $m/64 = 0.5$.

## 1.3 Hierarchical (two-level) metadata

Quantize the *scales themselves*. Let each super-block contain $K$ blocks of $G$ weights. Store per block a $b_s$-bit scale and $b_z$-bit zero; store per super-block one fp16 super-scale and super-zero ($32$ bits total).

**Definition 1.3 (Hierarchical effective rate).**
$$
\boxed{\;B^{\text{hier}}_{\text{eff}} = b + \frac{b_s + b_z}{G} + \frac{32}{KG}\;}
$$

**Example 1.4 (llama.cpp Q4_K, exactly).** $b = 4$, $b_s = b_z = 6$, $G = 32$, $K = 8$:
$$
B^{\text{hier}}_{\text{eff}} = 4 + \frac{12}{32} + \frac{32}{256} = 4 + 0.375 + 0.125 = 4.500
$$

This reproduces llama.cpp's documented figure to the digit. **The $4.5$ bpw operating point on slide 7 has been public since June 2023, and the public version achieves it at $G = 32$ granularity rather than $G = 64$.** §2.6 quantifies exactly what that costs OSTIR.

**Example 1.5 (an aggressive hierarchical point).** $b = 4$, $b_s = b_z = 4$, $G = 32$, $K = 16$:
$$
B_{\text{eff}} = 4 + \frac{8}{32} + \frac{32}{512} = 4.3125, \qquad \varphi = 7.25\%
$$

**Example 1.6 (symmetric, no zero-point).** $m = 16$: $B_{\text{eff}}(4, 32) = 4.5$ at $G = 32$ with a single-level scheme. Requires the group to be mean-centered or rotated first; see §2.7.

## 1.4 Recovering $N_{\text{slice}}$ from the deck

Panel bytes are $S = N B_{\text{eff}} / 8$. Inverting on Path A ($S = 2.154$ MiB, $B_{\text{eff}} = 5.0$):

$$
N = \frac{8 S}{B_{\text{eff}}} = \frac{8 \times 2.154 \times 2^{20}}{5.0} = 3{,}613{,}812 \text{ weights}
$$

**Proposition 1.7 (The scatter is exactly collinear).** With $N = 3.6138 \times 10^6$ fixed, the model $S(G) = N B_{\text{eff}}(4,G)/8$ predicts:

| $G$ | Predicted (MiB) | Deck (MiB) | Error |
|---|---|---|---|
| 32 | 2.1540 | 2.154 | 0.00% |
| 64 | 1.9386 | 1.939 | 0.02% |
| 128 | 1.8309 | 1.831 | 0.01% |
| 256 | 1.7771 | 1.777 | 0.01% |

All four points agree to within rounding. **The deck's slice model is internally consistent and reduces to a single free parameter.** $\blacksquare$

**Interpretation.** $N \approx 3.61 \times 10^6$ is a *GEMM panel*, not a model. It corresponds to tilings such as $m_c \times k_c = 2048 \times 1764$, $1792 \times 2016$, or $3584 \times 1008$. For scale: a Llama-70B-class MLP matrix ($4096 \times 14336 = 5.87 \times 10^7$ weights) is $16.2$ such panels — i.e. roughly one weight matrix per 16 cores.

## 1.5 The capacity form

Solving the residency constraint $N B_{\text{eff}}/8 \leq \eta C$ for $N$:

**Definition 1.8 (Capacity function).**
$$
\boxed{\;N_{\max}(b, G) = \frac{8\eta C}{B_{\text{eff}}(b,G)}\;}
$$

This is the object the entire thesis optimizes: *how many weights can be held resident per core.*

---

# Part II — Rate–Distortion Theory of Grouped Quantization

## 2.1 The calculus of the capacity function

**Proposition 2.1 (Monotonicity and concavity).** $N_{\max}$ is strictly increasing and strictly concave in $G$, with $\lim_{G \to \infty} N_{\max} = 8\eta C / b$.

*Proof.* Write $B(G) = b + m/G$, so $B'(G) = -m/G^2$. With $\mathcal{K} = 8\eta C$,
$$
\frac{dN_{\max}}{dG} = -\frac{\mathcal{K}}{B^2} B'(G) = \frac{\mathcal{K}m}{G^2 B(G)^2} > 0 .
$$
Differentiating again, the $G^{-4}$ and $G^{-3}$ terms give $d^2N_{\max}/dG^2 < 0$ for all $G > 0$. The limit follows from $B(G) \to b$. $\blacksquare$

Concavity is the mathematical statement of diminishing returns: **every doubling of $G$ buys less capacity than the last, while distortion grows without bound.**

## 2.2 The elasticity identity — what "the 11%" actually is

**Theorem 2.2 (Elasticity equals metadata fraction).**
$$
\boxed{\;\varepsilon \;\equiv\; \frac{d \ln N_{\max}}{d \ln G} \;=\; \frac{m/G}{B_{\text{eff}}} \;=\; \varphi\;}
$$

*Proof.*
$$
\varepsilon = \frac{G}{N_{\max}}\frac{dN_{\max}}{dG} = \frac{G B}{\mathcal{K}} \cdot \frac{\mathcal{K}m}{G^2B^2} = \frac{m}{GB} = \varphi . \qquad \blacksquare
$$

**Corollary 2.3.** At the deck's operating point $(b,G) = (4,64)$, $\varepsilon = \varphi = 1/9 = 11.11\%$.

This is the exact, closed-form meaning of "the 11%." It is *not* a compression target. It is the **marginal elasticity of resident capacity with respect to quantization granularity**, and it is simultaneously the entire fraction of the bit budget still occupied by metadata.

**Proposition 2.4 (Doubling lemma).** Doubling the group size expands capacity by exactly $\varphi(2G)$:
$$
\frac{N_{\max}(2G)}{N_{\max}(G)} - 1 = \frac{B(G) - B(2G)}{B(2G)} = \frac{m/(2G)}{B(2G)} = \varphi(2G) .
$$

*Check:* $G: 32 \to 64$ gives $5.0/4.5 - 1 = 11.11\% = \varphi(64)$. $\blacksquare$

**Theorem 2.5 (Exhaustion bound).** The *total* capacity available from eliminating all metadata at fixed $b$ is
$$
\frac{N_{\max}(b, \infty)}{N_{\max}(b,G)} - 1 = \frac{\varphi}{1 - \varphi} .
$$
At $\varphi = 1/9$: $12.5\%$. $\blacksquare$

**This is the single most important structural fact in Part II.** At the current operating point, *perfect, free, lossless elimination of every metadata bit* would buy $12.5\%$ more resident capacity. Realistic schemes (Ex. 1.5) recover about $4.3\%$. Metadata compression is essentially finished as a lever.

## 2.3 Where the range comes from: extreme-value theory

Under (A2)–(A3), the group range is a functional of the sample maximum. For $G$ i.i.d. standard normals,
$$
\mathbb{E}[\max_G] \approx \sqrt{2\ln G} - \frac{\ln\ln G + \ln 4\pi}{2\sqrt{2\ln G}}, \qquad \mathbb{E}[R_G] = 2\,\mathbb{E}[\max_G]\,\sigma .
$$

The asymptotic form underestimates at small $G$; the table below uses tabulated exact values.

**Table 2.1 — Grouped uniform quantization at $b = 4$, Gaussian weights**

| $G$ | $\mathbb{E}[R_G]/\sigma$ | $\Delta/\sigma = R/15$ | $D/\sigma^2 = \Delta^2/12$ | SQNR (dB) |
|---|---|---|---|---|
| 16 | 3.53 | 0.2354 | 0.004619 | 23.35 |
| 32 | 4.14 | 0.2757 | 0.006335 | 21.98 |
| 64 | 4.64 | 0.3095 | 0.007982 | 20.98 |
| 128 | 5.16 | 0.3437 | 0.009845 | 20.07 |
| 256 | 5.66 | 0.3775 | 0.011878 | 19.25 |

**Proposition 2.6 (Logarithmic distortion law).** Under (A2), $D(b,G) \propto \dfrac{\ln G}{(2^b-1)^2}$ asymptotically:
$$
D \approx \frac{\big(2\sigma\sqrt{2\ln G}\big)^2}{12(2^b-1)^2} = \frac{2\sigma^2 \ln G}{3(2^b-1)^2} .
$$

**Rate grows as $m/G$; distortion grows as $\ln G$.** The first is a hyperbola collapsing to zero; the second is unbounded. Their crossing point is the design.

## 2.4 The constrained program and its KKT conditions

Formally, the design problem is:

$$
\begin{aligned}
\min_{b,\,G,\,m_c,\,k_c} \quad & D(b,G) \\
\text{s.t.} \quad & \underbrace{\tfrac{1}{8} m_c k_c B_{\text{eff}}(b,G)}_{\text{weights}} + S_{\text{kv}} + S_{\text{act}} + S_{\text{scr}} \;\leq\; \eta C \\
& m_c k_c \geq N_{\min} \quad \text{(reuse floor, §3.5)} \\
& b \in \{2,3,4,5,6\}, \quad G \in \{16,32,64,128,256\}
\end{aligned}
$$

With $\lambda \geq 0$ the multiplier on the residency constraint, the Lagrangian relaxation over continuous $G$ gives
$$
\mathcal{L} = \frac{c\ln G}{(2^b-1)^2} + \lambda\left(b + \frac{m}{G}\right), \qquad c = \tfrac{2}{3}\sigma^2 ,
$$
$$
\frac{\partial \mathcal{L}}{\partial G} = \frac{c}{G(2^b-1)^2} - \frac{\lambda m}{G^2} = 0
\;\;\Longrightarrow\;\;
\boxed{\;G^\star = \frac{\lambda m (2^b-1)^2}{c}\;}
$$

**Theorem 2.7 (Optimal granularity scaling).** The optimal group size grows *linearly* in the shadow price of cache capacity and *quadratically* in the code range $(2^b - 1)$.

Two consequences worth internalizing:

- **Scarce cache $\Rightarrow$ coarse groups.** $\lambda$ is high exactly when residency binds hard; that is precisely when you should push $G$ up. The deck's instinct (go to $G = 64$) is correct, and this is why.
- **Higher bit width $\Rightarrow$ much coarser groups.** Because the dependence is quadratic, at $b = 6$ the optimal $G$ is $(63/15)^2 = 17.6\times$ larger than at $b = 4$. Fine-grained groups are only worth their metadata at *low* bit widths.

## 2.5 Bennett's integral and the Panter–Dite bound

For non-uniform (companded) quantization with point density $\lambda(x)$, $\int \lambda = 1$, and $N_\ell = 2^b$ levels, the local step is $\Delta(x) \approx [N_\ell \lambda(x)]^{-1}$, giving **Bennett's integral**:
$$
D = \int f(x)\frac{\Delta(x)^2}{12}dx = \frac{1}{12 N_\ell^2}\int \frac{f(x)}{\lambda(x)^2}dx .
$$

Minimize by calculus of variations with the constraint $\int\lambda = 1$. Stationarity of $\int [f\lambda^{-2} + \mu\lambda]\,dx$ gives $-2f\lambda^{-3} + \mu = 0$, hence

$$
\boxed{\;\lambda^\star(x) = \frac{f(x)^{1/3}}{\int f(u)^{1/3}du}\;}
$$

Substituting yields the **Panter–Dite formula**:
$$
D^\star = \frac{1}{12 N_\ell^2}\left(\int f(x)^{1/3}dx\right)^3 = \frac{2^{-2b}}{12}\left(\int f^{1/3}\right)^3 .
$$

For $f = \mathcal{N}(0,\sigma^2)$, $\left(\int f^{1/3}\right)^3 = \tfrac{3\sqrt{3}\,\pi}{...}\,\sigma^2 = 32.65\,\sigma^2$, so
$$
\boxed{\;D^\star_{\text{Gauss}} = \frac{\sqrt{3}\pi}{2}\,\sigma^2 2^{-2b} = 2.7207\,\sigma^2 2^{-2b}\;}
$$

At $b=4$: $D^\star = 0.010628\,\sigma^2$.

## 2.6 The equal-rate theorem, and a warning about Lloyd–Max

**Theorem 2.8 (Hierarchical dominance at equal rate).** A two-level scheme achieving $B_{\text{eff}} = 4.5$ at $G = 32$ has lower distortion than a flat scheme achieving $B_{\text{eff}} = 4.5$ at $G = 64$, by
$$
\frac{D(4,64)}{D(4,32)} = \frac{0.007982}{0.006335} = 1.260 \quad\Longrightarrow\quad \Delta\text{SQNR} = 10\log_{10}1.260 = 1.00\text{ dB} .
$$
The second-order penalty from quantizing the scales to $b_s$ bits contributes $D_s/D \approx 0.2\%$ at $b_s = 6$ and is negligible. $\blacksquare$

**So: at identical $4.5$ bits/weight, the public GGUF scheme is $\approx 21\%$ lower MSE ($1.0$ dB) than the deck's Path B.** This is not a marginal difference and a technical diligence reviewer will find it. The fix is trivial — adopt hierarchical metadata — but the *claim* must change accordingly.

**Now the warning.** Compare Table 2.1 against Panter–Dite. At $b=4$:

| Scheme | $D/\sigma^2$ |
|---|---|
| Lloyd–Max / optimal companded, **global** | 0.010628 |
| Uniform min–max, **grouped at $G=64$** | 0.007982 |
| Uniform min–max, **grouped at $G=32$** | 0.006335 |

**Theorem 2.9 (Grouping subsumes companding).** Under (A2), per-group uniform quantization at $G \leq 64$ achieves strictly lower distortion than a globally optimal non-uniform quantizer at the same $b$.

*Reason.* Panter–Dite optimizes one codebook against the *full* marginal density including its tails. Grouping conditions on a sample of $G$ draws, whose empirical support is $\mathcal{O}(\sqrt{\ln G})$ rather than unbounded. Per-group adaptivity captures the adaptivity that companding was invented to supply. $\blacksquare$

**Consequence: the deck's emphasis on Lloyd–Max is misplaced.** Under Gaussian weights, a Lloyd–Max codebook buys nothing over grouped min–max — it *loses*. Engineering effort spent on codebook construction is effort not spent on the residency problem, which §4 shows is where all the leverage is.

## 2.7 Where non-uniform coding *does* pay: outliers

Theorem 2.9 rests on (A2). LLM weight and KV distributions violate it in a specific, structured way: a small fraction of channels carry magnitudes many $\sigma$ above the rest.

**Proposition 2.10 (Outlier catastrophe).** Let a group of $32$ contain $31$ draws from $\mathcal{N}(0,\sigma^2)$ and one value at $10\sigma$. Then $R \approx 20\sigma$, $\Delta = 1.333\sigma$, and
$$
D = \Delta^2/12 = 0.1481\,\sigma^2 ,
$$
a $23.4\times$ degradation versus the clean case ($0.006335\sigma^2$), equal to $-13.7$ dB. $\blacksquare$

**Proposition 2.11 (Outlier extraction is nearly free).** Store the top $p$ fraction of weights by magnitude as fp16 with a 16-bit index ($32$ bits each) and exclude them from group statistics. The rate penalty is
$$
\Delta B = 32p \text{ bits/weight}.
$$

| $p$ | $\Delta B$ | Rate penalty at $B=4.5$ |
|---|---|---|
| 0.05% | 0.016 | 0.36% |
| 0.1% | 0.032 | 0.71% |
| 0.5% | 0.160 | 3.6% |
| 1.0% | 0.320 | 7.1% |

**Design rule.** Operate at $p \in [0.05\%, 0.2\%]$. This costs under $1\%$ of the bit budget and removes the dominant distortion term. **Outlier handling, not codebook shape, is where the accuracy lives.**

## 2.8 Optimal clipping

Min–max is not the optimal range. Splitting distortion into granular and overload terms for a clip at $\pm\alpha\sigma$:
$$
D(\alpha) = \underbrace{\frac{(2\alpha\sigma)^2}{12\cdot 4^{b}}}_{\text{granular}} + \underbrace{2\int_{\alpha\sigma}^{\infty}(x-\alpha\sigma)^2 f(x)\,dx}_{\text{overload}}
$$
Setting $dD/d\alpha = 0$ yields a transcendental condition; the standard Gaussian solutions are

| $b$ | $\alpha^\star/\sigma$ |
|---|---|
| 2 | 1.71 |
| 3 | 2.15 |
| 4 | 2.55 |
| 8 | 3.92 |

Compare against min–max: at $G = 32$, $\mathbb{E}[\max] = 2.07\sigma$; at $G = 64$, $2.32\sigma$. **At $b=4$, min–max over $G \in [32,64]$ lands within $10\%$ of optimal clipping by accident.** At larger $G$ or heavier tails it does not. Use optimal clipping (with outliers extracted first) once $G \geq 128$.

---

# Part III — The Residency Constraint

## 3.1 Working-set decomposition

The per-core L2 must simultaneously hold:

$$
W = \underbrace{S_w}_{\text{weight panel}} + \underbrace{S_{kv}}_{\text{KV working set}} + \underbrace{S_{act}}_{\text{activations}} + \underbrace{S_{scr}}_{\text{dequant scratch}} + \underbrace{S_{str}}_{\text{streaming operand lines}} \;\leq\; \eta C
$$

**The deck models $W \approx S_w$ and $\eta \approx 1$.** Both are wrong, and the second is wrong by a large factor.

## 3.2 The usable-capacity factor $\eta$

L2 is set-associative (typically 16-way on Xeon 6 / Sapphire Rapids, $C = 2$ MiB). Three effects prevent $\eta \to 1$:

1. **Conflict misses.** Non-power-of-two panel strides collide in sets. Mitigated by padded packing.
2. **Streaming pollution.** The $B$ operand and output tiles stream through L2 and evict panel lines unless non-temporal stores or explicit prefetch hints are used.
3. **Shared occupancy.** Stack, thread-local state, and the dequantized tile all reside concurrently.

**Established practice (BLIS/GotoBLAS) allocates $50\%$–$70\%$ of L2 to the resident panel.** Take $\eta \in [0.5, 0.75]$.

**Proposition 3.1 (The deck's implied $\eta$ is infeasible).** Path B occupies $1.939$ of $2.000$ MiB, implying $\eta = 0.970$. At the realistic $\eta = 0.60$, the admissible panel is $1.200$ MiB and
$$
N_{\max} = \frac{8 \times 0.60 \times 2^{21}}{4.5} = 2{,}236{,}962 \text{ weights},
$$
which is $38\%$ smaller than the deck's $N_{\text{slice}} = 3.614\times10^6$. $\blacksquare$

**This is the real correction, and it is much larger than $11\%$.** The deck is over-subscribed by $38\%$, not under-compressed by $11\%$.

## 3.3 KV cache growth

For one layer, $h_{kv}$ KV heads assigned to a core, head dimension $d_h$, context $L$, KV rate $B_{kv}$:
$$
S_{kv}(L) = \frac{2\,L\,h_{kv}\,d_h\,B_{kv}}{8} \text{ bytes}.
$$
With OSTIR's stated 3-bit KV at $G_{kv} = 64$: $B_{kv} = 3 + 32/64 = 3.5$ bits.

**Table 3.1 — KV footprint, one layer, one head, $d_h = 128$, $B_{kv} = 3.5$**

| $L$ | $S_{kv}$ | % of 2 MiB L2 |
|---|---|---|
| 1,024 | 0.109 MiB | 5.5% |
| 4,096 | 0.438 MiB | 21.9% |
| 8,192 | 0.875 MiB | 43.8% |
| 32,768 | 3.500 MiB | 175% |

**Theorem 3.2 (KV critical context).** The context at which KV alone consumes fraction $\beta$ of L2 is
$$
\boxed{\;L^\star = \frac{8\,\beta\,C}{2\,h_{kv}\,d_h\,B_{kv}}\;}
$$
At $\beta = 0.25$, $C = 2^{21}$, $h_{kv} = 1$, $d_h = 128$, $B_{kv} = 3.5$: $L^\star = 4{,}681$ tokens. $\blacksquare$

**Theorem 3.3 (Co-residency impossibility).** Under $\eta = 0.60$ and $C = 2$ MiB, weights ($1.939$ MiB) and KV ($\geq 0.438$ MiB at $L \geq 4096$) cannot be simultaneously L2-resident for any context length of practical interest. Indeed $1.939 > \eta C = 1.200$ alone.

*Corollary.* **The architecture must time-multiplex L2 between a GEMM phase and an attention phase.** There is no static allocation that satisfies both. $\blacksquare$

## 3.4 The correct reformulation: blocked GEMM

Theorem 3.3 is not fatal — it is a redirection. The right framework is the classical Goto/BLIS cache-blocking structure, in which the L2-resident object is the **packed $A$-panel of the current GEMM phase**, not the model and not the KV cache.

For $C \mathrel{+}= A B$ with $A \in \mathbb{R}^{M\times K}$, $B \in \mathbb{R}^{K\times N_o}$, and an L2-resident $A$-panel of $m_c \times k_c$:

$$
\text{Traffic} \;\approx\; \underbrace{MK\frac{B_{\text{eff}}}{8}}_{A \text{ read once}} \;+\; \underbrace{K N_o s_B \left\lceil \frac{M}{m_c} \right\rceil}_{B \text{ re-read per panel row}} \;+\; \underbrace{2 M N_o s_C}_{C}
$$

Substituting the residency-limited panel $m_c = 8\eta C'/(k_c B_{\text{eff}})$:

$$
\left\lceil \frac{M}{m_c} \right\rceil = \frac{M k_c B_{\text{eff}}}{8\eta C'}
$$

**Theorem 3.4 (Traffic is strictly increasing in bit rate — twice).**
$$
\text{Traffic}(B_{\text{eff}}) = \frac{MK}{8}B_{\text{eff}} + \frac{K N_o s_B M k_c}{8\eta C'}B_{\text{eff}} + 2MN_o s_C
$$
$$
\frac{\partial\,\text{Traffic}}{\partial B_{\text{eff}}} = \frac{MK}{8} + \frac{K N_o s_B M k_c}{8\eta C'} \;>\; 0
$$
The second term dominates whenever $N_o s_B k_c / (\eta C') > 1$, i.e. for wide outputs. $\blacksquare$

**This theorem is the mathematical heart of the OSTIR thesis and the correct basis for its IP claim.** Compression reduces DRAM traffic *twice*: directly, because weights are smaller; and indirectly and usually more importantly, because a smaller bit rate permits a larger resident panel, which reduces how many times the streaming operand must be re-read. The second channel is invisible in the deck and is the more defensible contribution.

## 3.5 The reuse floor

Panels cannot shrink indefinitely. The re-read term grows as $1/m_c$, so there is a floor $N_{\min}$ below which repacking and re-streaming dominate. Empirically this is where $\lceil M/m_c \rceil$ exceeds $\approx 8$. Add as a constraint (already stated in §2.4).

## 3.6 The strategic consequence

Collecting Theorems 2.5, 3.1, and 3.4:

| Lever | Capacity gained | Cost | Verdict |
|---|---|---|---|
| Metadata compression (all of it) | $+12.5\%$ | accuracy, complexity | **Exhausted** |
| Realistic metadata scheme (Ex. 1.5) | $+4.3\%$ | small | Marginal |
| $b: 4 \to 3$ | $+28.6\%$ | $\approx 6$ dB SQNR | Expensive |
| **Re-tiling $k_c$ downward** | **unbounded** | **more $B$ re-reads** | **The actual lever** |
| Outlier extraction | $-0.7\%$ | negligible | Do it anyway (accuracy) |

**The prototype should not chase 11%. It should re-solve for $(m_c, k_c)$ against a measured $\eta$.** Tiling is free, continuous, and unbounded; compression is none of those.

---

# Part IV — The Performance Model

## 4.1 Roofline setup

Per-core AMX INT8 peak: `TDPBSSD` computes $(m,n,k) = (16,16,64)$, i.e. $16{,}384$ MACs per instruction, at $\approx 16$ cycles throughput $\Rightarrow \pi = 2048$ ops/cycle.

Bandwidths (deck's own figures, which are defensible): $100$ GB/s L2 and $8.5$ GB/s DRAM per core at $\approx 3$ GHz give
$$
\beta_{L2} \approx 33\ \text{B/cycle}, \qquad \beta_{DR} \approx 2.83\ \text{B/cycle}, \qquad r = \frac{\beta_{DR}}{\beta_{L2}} = 0.085 .
$$

*Note on the deck's "$\sim$90 ns":* on re-reading slide 7, the $90$ ns label sits on the **miss $\to$ DRAM** edge, where it is correct (DRAM load-to-use latency). L2 hit latency is $\approx 16$ cycles ($\approx 5$ ns). The diagram is defensible as drawn; do not let a reviewer read the $90$ ns as an L2 figure, and label the edge explicitly.

## 4.2 Arithmetic intensity of decode

At batch 1, each weight is loaded once and used in one MAC (2 ops):
$$
I_{\text{dec}} = \frac{2\ \text{ops}}{B_{\text{eff}}/8\ \text{bytes}} = \frac{2}{0.5625} = 3.56\ \text{ops/byte}.
$$

Ridge points $I^\star = \pi/\beta$:

| Source | $\beta$ (B/cyc) | $I^\star$ | AMX utilization at $I = 3.56$ |
|---|---|---|---|
| DRAM | 2.83 | 724 | **0.49%** |
| L2 | 33 | 62 | **5.7%** |

**Theorem 4.1 (AMX is irrelevant at batch 1).** Batch-1 decode operates at under $6\%$ of AMX peak even with a perfectly L2-resident panel. Assumption (A4) holds by a margin of $17\times$. $\blacksquare$

**Corollary (the batch threshold).** With batching, each loaded weight serves $n_b$ MACs, so $I(n_b) = 2n_b/0.5625$. Compute-bound from L2 requires $I \geq 62$:
$$
\boxed{\;n_b^\star = \frac{62 \times 0.5625}{2} \approx 18\;}
$$

**This is a genuinely load-bearing result for the business.** The L2-residency architecture pays off in the AMX regime only at batch $\gtrsim 18$. Agentic burst traffic — many concurrent tool-calling sessions sharing prefixes — naturally produces exactly this batch profile. Interactive single-user chat does not. **The ICP and the physics agree, and the deck should say so.**

## 4.3 The residency law

Let $h$ be the fraction of weight-stream bytes served from L2. Effective bandwidth composes harmonically:
$$
\frac{1}{\beta_{\text{eff}}} = \frac{h}{\beta_{L2}} + \frac{1-h}{\beta_{DR}}
$$

**Theorem 4.2 (Residency law).**
$$
\boxed{\;S(h) = \frac{\beta_{\text{eff}}}{\beta_{DR}} = \frac{1}{1 - h(1-r)}\;}, \qquad r = \frac{\beta_{DR}}{\beta_{L2}}
$$

*Proof.* $\beta_{\text{eff}} = [h/\beta_{L2} + (1-h)/\beta_{DR}]^{-1}$. Divide by $\beta_{DR}$ and substitute $r$. $\blacksquare$

**Table 4.1 — $S(h)$ at $r = 0.085$**

| $h$ | $S(h)$ |
|---|---|
| 1.000 | **11.76** |
| 0.990 | 10.62 |
| 0.984 | **10.00** |
| 0.950 | 7.65 |
| 0.911 | 6.00 |
| 0.900 | 5.67 |
| 0.750 | 3.19 |
| 0.500 | 1.84 |
| 0.000 | 1.00 |

**Corollary 4.3.** The deck's $12\times$ is exactly $S(1)$: it is the *asymptotic ceiling at perfect residency*, not an operating point.

**Sensitivity.**
$$
\frac{dS}{dh} = \frac{1-r}{[1-h(1-r)]^2} = (1-r)S(h)^2
$$
At $h = 0.95$: $dS/dh = 0.915 \times 7.65^2 = 53.5$. **One percentage point of hit rate is worth $0.54\times$ of speedup.** At $h = 0.99$ it is worth $1.03\times$.

**This inverts the engineering priority.** Recovering the entire $11\%$ metadata budget (Thm. 2.5, worth $12.5\%$ capacity) is worth far less than moving $h$ from $0.95$ to $0.99$. Chase misses, not bits.

## 4.4 End-to-end composition

Weight traffic is not all traffic. Let $f$ be the weight share of per-token decode bytes.

$$
\text{Weight bytes/token} = N_{\text{params}}\frac{B_{\text{eff}}}{8}, \qquad
\text{KV bytes/token} = \frac{2 L n_\ell h_{kv} d_h B_{kv}}{8}
$$

**Worked example — 7B class, $L = 8192$, $n_\ell = 32$, $d_h = 128$, $B_{\text{eff}} = 4.5$, $B_{kv} = 3.5$:**

| Attention | $h_{kv}$ | Weights | KV | $f$ |
|---|---|---|---|---|
| MHA | 32 | 3.94 GB | 0.94 GB | 0.807 |
| GQA-8 | 8 | 3.94 GB | 0.235 GB | 0.944 |

**Corollary 4.4 (Amdahl composition).**
$$
\boxed{\;S_{\text{total}} = \left[\frac{f}{S_w} + \frac{1-f}{S_{kv}}\right]^{-1}\;}
$$
With perfect weight residency ($S_w = 11.76$) and no KV acceleration ($S_{kv} = 1$):

| Case | $f$ | $S_{\text{total}}$ |
|---|---|---|
| MHA | 0.807 | **3.86×** |
| GQA-8 | 0.944 | **7.15×** |

**Two conclusions the deck must absorb.**

1. **The honest end-to-end number is 4–7×, not 12×.** $12\times$ is the weight-stream ceiling at $h=1$; $4$–$7\times$ is what a customer measures. $7\times$ on hardware they already own is still a compelling claim — arguably more compelling, because it survives diligence.
2. **GQA is worth $1.85\times$ end-to-end by itself.** Model selection is a first-class performance lever, on par with the entire quantization program. Target GQA/MQA architectures preferentially.

---

# Part V — Algorithms

## 5.1 The group-size solver (the control loop)

This is the routine that constitutes the defensible novelty (Part VII).

```
ALGORITHM 1 — SolveResidentConfig
INPUT:  measured C (L2 bytes/core), measured η, accuracy floor D_max,
        layer dims (M, K, N_o), KV profile (L, n_ℓ, h_kv, d_h)
OUTPUT: (b, G, m_c, k_c, p_outlier)

1.  C' ← η · C − S_act − S_scr − S_str          # measured, not assumed
2.  P ← ∅                                       # Pareto candidates
3.  for b in {3,4,5}:
4.      for G in {16,32,64,128,256}:
5.          B ← b + m/G                          # or hierarchical, Def 1.3
6.          D ← EstimateDistortion(b,G,p)        # Table 2.1 / calibration
7.          if D > D_max: continue
8.          N_max ← 8·C'/B                       # Def 1.8
9.          for k_c in divisors(K):
10.             m_c ← floor(N_max / k_c)
11.             if m_c·k_c < N_min: continue     # reuse floor §3.5
12.             T ← ModelTraffic(M,K,N_o,m_c,k_c,B)   # Thm 3.4
13.             P ← P ∪ {(b,G,m_c,k_c,D,T)}
14. return argmin_{P} T subject to D ≤ D_max
```

**Note line 1 and line 12.** The loop measures the machine, then minimizes *modeled DRAM traffic* — not bit rate — subject to an accuracy floor. This is what distinguishes it from published quantizers, which minimize bits or distortion without a hardware constraint.

## 5.2 Hierarchical quantizer

```
ALGORITHM 2 — HierarchicalQuantize(W, G, K, b, b_s, b_z)
1.  split W into super-blocks of K·G weights
2.  for each super-block SB:
3.      for each block j in SB:
4.          s_j ← (max(B_j) − min(B_j)) / (2^b − 1)
5.          z_j ← min(B_j)
6.      S_max ← max_j s_j ;  Z_range ← (max_j z_j − min_j z_j)
7.      ŝ_j ← round(s_j / S_max · (2^{b_s}−1))         # quantized scale
8.      ẑ_j ← round((z_j − min_j z_j)/Z_range · (2^{b_z}−1))
9.      store fp16 S_max, fp16 min_j z_j                # 32 bits/super-block
10.     for each w_i in block j:
11.         q_i ← clamp(round((w_i − ẑ_j·Δz − z_min)/(ŝ_j·ΔS)), 0, 2^b−1)
12. rate = b + (b_s+b_z)/G + 32/(K·G)                    # Def 1.3
```

Validate against Ex. 1.4: $(b,b_s,b_z,G,K) = (4,6,6,32,8) \Rightarrow 4.500$ exactly.

## 5.3 Outlier extraction

```
ALGORITHM 3 — ExtractOutliers(W_tile, p)
1.  τ ← quantile(|W_tile|, 1 − p)
2.  O ← {(i, w_i) : |w_i| > τ}                  # fp16 value + uint16 index
3.  W' ← W_tile with O positions set to group mean
4.  return HierarchicalQuantize(W'), O
5.  # rate penalty: 32·p bits/weight (Prop 2.11)
6.  # dequant: reconstruct dense, then scatter O over it (branch-free)
```

Run **before** computing group min/max — that is the entire point (Prop. 2.10).

## 5.4 LUT dequantization on AVX-512

A frequent objection is that non-uniform codebooks are too slow on CPU. For $b = 4$ this is false.

A 16-entry codebook fits one byte-shuffle lane. `vpshufb` / `vpermb` on a 512-bit register performs **64 parallel lookups per instruction**. The only real cost is reloading the table per group.

**Proposition 5.1 (Codebook amortization).** Let table reload cost $T_r \approx 3$ cycles and shuffle throughput $64$ weights/instruction. Per-weight overhead of a *per-group* codebook is $T_r/G$ cycles.

| $G$ | overhead (cyc/weight) |
|---|---|
| 32 | 0.094 |
| 64 | **0.047** |
| 128 | 0.023 |

At $G \geq 64$ the overhead is under $0.05$ cycles/weight — negligible against the $\approx 0.28$ cycles/weight DRAM cost (§4.2). **Per-group non-uniform codebooks are affordable at $b=4$.** They are simply not *useful* under (A2), per Thm. 2.9 — reserve them for outlier-heavy layers flagged by calibration.

## 5.5 The KV policy engine, and a surprising result

The runtime must choose among **reuse** (fetch cached KV from a tier), **transfer** (pull from a peer), and **recompute** (re-run prefill over the prefix).

For prefix length $L_p$:
$$
T_{\text{fetch}} = \frac{L_p \cdot \text{bytes/token}}{\beta_{\text{tier}}}, \qquad
T_{\text{recompute}} = \frac{L_p \cdot \text{flops/token}}{\pi_{\text{eff}}}
$$

**Theorem 5.2 (Recompute crossover bandwidth).** Recompute beats fetch iff
$$
\boxed{\;\beta_{\text{tier}} < \beta^\star = \pi_{\text{eff}} \cdot \frac{\text{bytes/token}}{\text{flops/token}}\;}
$$

**Theorem 5.3 (On CPU, never recompute).** For a 7B GQA-8 model: bytes/token $= 2 \times 32 \times 8 \times 128 \times 3.5/8 = 28{,}672$ B; flops/token $\approx 2 \times 7\times10^9 = 1.4\times10^{10}$. With $\pi_{\text{eff}} = 4.1$ Tops/s per core:
$$
\beta^\star = 4.1\times10^{12} \times \frac{28{,}672}{1.4\times10^{10}} = 8.4\ \text{MB/s}
$$
Every storage tier — DRAM ($\sim$10 GB/s), NVMe ($\sim$3 GB/s), even network ($\sim$1 GB/s) — exceeds $\beta^\star$ by two to three orders of magnitude. $\blacksquare$

*Concretely, at $L_p = 1000$:* fetch from NVMe $= 28.7$ MB $/\,3$ GB/s $= 9.6$ ms. Recompute $= 1.4\times10^{13}/4.1\times10^{12} = 3.4$ s. **Fetch wins by $355\times$.**

**This is the opposite of the GPU regime**, where a $10^3$ Tflop/s peak moves $\beta^\star$ into the tens of GB/s and recompute becomes genuinely competitive. On CPU it never is.

**Design consequence: collapse the three-way policy to two-way (reuse vs. transfer) and always persist KV.** Spend the saved complexity on prefix-tree hit rate — which, by Thm. 4.2, is where the returns are anyway.

---

# Part VI — Validation Protocol

Everything above is a model. This part makes it falsifiable on one socket.

## 6.1 Environment

```bash
# Capability check — AMX requires Sapphire Rapids or later
lscpu | grep -o 'amx_tile\|amx_int8\|amx_bf16\|avx512_vnni'
lscpu -C                      # confirm L1d/L2/L3 sizes and associativity

# AMX requires explicit permission from the kernel (XFD)
#   arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)

# Reproducibility: pin frequency, disable deep C-states, isolate cores
sudo cpupower frequency-set -g performance
sudo cpupower idle-set -D 0
# kernel cmdline: isolcpus=8-15 nohz_full=8-15 rcu_nocbs=8-15
echo 1024 | sudo tee /proc/sys/vm/nr_hugepages
```

**Statistical discipline:** $n \geq 30$ runs per configuration; report **median and IQR**, never mean; discard the first 3 runs (warm-up); pin with `taskset -c` and `numactl --membind`.

## 6.2 Counters

| Quantity | Event |
|---|---|
| L2 hits | `MEM_LOAD_RETIRED.L2_HIT` |
| L2 misses | `MEM_LOAD_RETIRED.L2_MISS` |
| Lines filled | `L2_LINES_IN.ALL` |
| Demand misses | `L2_RQSTS.ALL_DEMAND_MISS` |
| Offcore reads | `OFFCORE_REQUESTS.ALL_DATA_RD` |
| Stall cycles | `CYCLE_ACTIVITY.STALLS_L2_MISS` |

$$
\hat{h} = \frac{\texttt{L2\_HIT}}{\texttt{L2\_HIT} + \texttt{L2\_MISS}}
$$

Cross-check with TMA: `toplev -l3` should attribute the shift as `Backend_Bound.Memory_Bound.DRAM_Bound` $\to$ `L2_Bound`.

## 6.3 The six experiments

---

### **E1 — Bit-rate verification** *(1 day)*
**Tests:** Def. 1.1, Def. 1.3, Prop. 1.7.
**Method:** implement flat and hierarchical quantizers; serialize; compare on-disk bytes to $N B_{\text{eff}}/8$ for $G \in \{16,32,64,128,256\}$, $b \in \{3,4\}$.
**Pass:** measured $=$ predicted within $0.1\%$; Q4_K config reproduces $4.500$ exactly.
**Falsifies if:** the algebra is misimplemented. Cheap and non-negotiable — everything downstream inherits it.

---

### **E2 — Residency knee** *(2 days)*
**Tests:** Prop. 3.1 — the true $\eta$.
**Method:** synthetic blocked GEMV; sweep $m_c k_c$ from $0.25$ MiB to $3$ MiB in $32$ steps at fixed $B_{\text{eff}} = 4.5$; record $\hat{h}$ at each.
**Output:** the knee $N^\dagger$ where $\hat{h}$ collapses. Then $\hat\eta = N^\dagger B_{\text{eff}}/(8C)$.
**Prediction:** $\hat\eta \in [0.5, 0.75]$, i.e. knee at $1.0$–$1.5$ MiB — **not** at $1.94$ MiB.
**Pass:** knee located within $10\%$ across 3 repetitions.
**This experiment alone settles whether the deck's Path B is feasible.**

---

### **E3 — Bandwidth constants** *(1 day)*
**Tests:** the inputs to Thm. 4.2.
**Method:** Intel MLC (`mlc --max_bandwidth`, `mlc --latency_matrix`) plus a hand-rolled per-core streaming read sized to fit L2 and to exceed L3.
**Output:** $\beta_{L2}$, $\beta_{DR}$, hence $r$.
**Prediction:** $r \in [0.06, 0.12]$; the deck's $0.085$ should land inside.
**Pass:** three runs within $5\%$.

---

### **E4 — Residency-law validation** ★ **THE CRITICAL EXPERIMENT** *(1 week)*
**Tests:** Thm. 4.2 — the entire performance thesis.
**Method:** generate $\geq 20$ configurations spanning $h \in [0.3, 1.0]$ by varying $m_c k_c$ around the E2 knee. For each, measure $\hat h$ (counters) and tok/core-sec. Fit
$$
S(\hat h) = \frac{1}{1 - \hat h(1-r)}
$$
with $r$ **fixed from E3** — no free parameters.
**Pass:** $R^2 > 0.90$ and residuals unstructured in $\hat h$.
**Fail modes and what they mean:**
- Systematic overprediction at high $h$ $\Rightarrow$ a serialization bottleneck (dequant, packing) not in the model; add a compute term.
- $S$ saturating below $1/r$ $\Rightarrow$ $\beta_{L2}$ is not achievable in the real kernel; re-measure with the actual access pattern.
- No fit at all $\Rightarrow$ the bandwidth model is wrong and Part IV must be rebuilt from measurement.

**If E4 fails, do not raise on the 12×.** Everything in the deck's slide 7 depends on this fit.

---

### **E5 — Accuracy frontier** *(1 week)*
**Tests:** Table 2.1, Thm. 2.8, Prop. 2.10, Prop. 2.11.
**Method:** for $b \in \{3,4\}$, $G \in \{16,32,64,128,256\}$, flat vs. hierarchical, $p_{\text{outlier}} \in \{0, 0.001, 0.005\}$ — measure WikiText-2 perplexity, MMLU, GSM8K, and a long-context retrieval probe.
**Predictions to check:**
- Hierarchical @ $G{=}32$, $4.5$ bpw beats flat @ $G{=}64$, $4.5$ bpw by $\approx 1.0$ dB SQNR and a visible $\Delta$PPL (**Thm. 2.8**).
- Outlier extraction at $p = 0.1\%$ recovers most of the gap between $G{=}64$ and $G{=}32$ (**Prop. 2.10**).
- Lloyd–Max codebooks at $G \leq 64$ do **not** beat grouped min–max (**Thm. 2.9**) — if they do, (A2) is violated in your models and Part II needs a heavier-tailed prior.
**Pass:** $\Delta\text{PPL} < 0.15$ vs. fp16 at the chosen operating point.

---

### **E6 — Batch threshold** *(3 days)*
**Tests:** Cor. to Thm. 4.1 — $n_b^\star \approx 18$.
**Method:** sweep batch $1 \to 64$; measure tok/s and AMX utilization (`EXE_ACTIVITY` / TMA `Core_Bound`).
**Pass:** the memory-bound $\to$ compute-bound transition observed within $\pm 40\%$ of $n_b = 18$.
**Business meaning:** confirms the ICP. If the knee is at $n_b = 4$, single-session workloads are viable and the addressable market widens. If it is at $n_b = 60$, only heavy multi-tenant burst traffic works, and the GTM must narrow accordingly.

---

### **E7 — End-to-end Amdahl** *(3 days)*
**Tests:** Cor. 4.4.
**Method:** instrument per-token byte traffic split (weights vs. KV vs. activations) to obtain $\hat f$; predict $S_{\text{total}}$; compare to measured end-to-end speedup vs. an unquantized DRAM-resident baseline.
**Prediction:** $S_{\text{total}} \in [3.5, 7.5]$ depending on GQA.
**Pass:** predicted within $20\%$ of measured.
**Deliverable:** this is the number that goes in the deck.

---

## 6.4 Sequencing and decision gates

```
Week 1:  E1, E3          → algebra + constants locked
Week 2:  E2              → GATE: is η ≥ 0.5? If η < 0.4, re-tile before proceeding.
Week 3:  E4              → GATE: does R² > 0.90? If not, STOP and rebuild Part IV.
Week 4:  E6, E7          → GATE: is S_total ≥ 3×? If not, the thesis is not investable as stated.
Week 5:  E5              → accuracy operating point chosen
Week 6:  Publish
```

**File the provisional patent before Week 6.** Publication starts the §102 clock: the US grants a one-year grace period, but the EPO and CNIPA apply absolute novelty and a pre-filing disclosure destroys those rights permanently.

---

# Part VII — What Is Actually Novel

Separating what is published from what is not:

| Component | Status |
|---|---|
| Grouped affine quantization, $b + m/G$ | Public — universal |
| Hierarchical metadata / double quantization ($4.5$ bpw) | **Public since 2023** — GGUF K-quants, QLoRA |
| Lloyd–Max / non-uniform codebooks for LLMs | Public — SqueezeLLM, KVQuant |
| Sub-4-bit KV, per-channel keys | Public — KIVI, KVQuant |
| Ternary packing | Public — TQ1_0/TQ2_0, BitNet |
| Cache blocking for GEMM | Public since 2008 — Goto & van de Geijn |
| Roofline analysis | Public since 2009 |
| **Closing the loop: measure $(C,\eta)$ at runtime, solve for $(b,G,m_c,k_c)$ minimizing modeled DRAM traffic subject to an accuracy floor, verify residency via hardware counters, and adapt** | **Not located in the literature** |
| **Thm. 3.4: bit rate governs traffic twice — directly and through panel size $\to$ operand re-reads** | **Not located as an explicit design principle** |
| **Thm. 5.3: the CPU recompute crossover ($\beta^\star \approx 8$ MB/s) collapses the KV policy to two-way** | **Not located** |

**The claim to pursue is the control loop, not the quantizer.** Draft it as a machine improvement, not a mathematical method: recite the measured cache capacity, the named performance counters, the residency verification step, and the resulting reduction in measured DRAM traffic. Under the *Enfish*/*Koninklijke KPN* line and the 2024 USPTO AI guidance, that framing survives §101; "select $G$ to minimize bits" does not.

**Design-around note.** Amazon US12,093,806 B1 claims a compiler that statically partitions a network so each subgraph fits a processing unit's dedicated cache. OSTIR's differentiator must be explicit and claimed: the knob is **quantization granularity**, not graph partitioning, and the loop is **runtime and measurement-driven**, not compile-time and static. Keep those two distinctions in every claim.

---

# Appendix A — Consolidated Results

| # | Result | Statement |
|---|---|---|
| 1.7 | Collinearity | Deck scatter is exact with $N = 3.6138\times10^6$ |
| 2.2 | Elasticity | $\varepsilon = \varphi$; at $(4,64)$, $= 11.11\%$ |
| 2.4 | Doubling | $G \to 2G$ buys exactly $\varphi(2G)$ |
| 2.5 | Exhaustion | All metadata removal $= +12.5\%$ capacity |
| 2.7 | Granularity | $G^\star \propto \lambda m (2^b-1)^2$ |
| 2.8 | Equal rate | Hierarchical beats flat by $1.00$ dB at $4.5$ bpw |
| 2.9 | Subsumption | Grouping $\succ$ global companding under (A2) |
| 2.10 | Outliers | One $10\sigma$ value costs $13.7$ dB |
| 3.1 | Feasibility | Deck implies $\eta = 0.97$; real $\eta \approx 0.6$; over-subscribed $38\%$ |
| 3.2 | KV critical | $L^\star = 8\beta C/(2h_{kv}d_h B_{kv}) = 4681$ tokens |
| 3.3 | Impossibility | Weights and KV cannot be co-resident |
| 3.4 | Traffic | Traffic strictly increasing in $B_{\text{eff}}$, through two channels |
| 4.1 | AMX | $<6\%$ utilization at batch 1; $n_b^\star \approx 18$ |
| 4.2 | Residency law | $S(h) = [1-h(1-r)]^{-1}$; $h \geq 0.984$ for $10\times$ |
| 4.4 | Amdahl | $S_{\text{total}} = 3.9\times$ (MHA) to $7.2\times$ (GQA-8) |
| 5.1 | LUT | Codebook overhead $< 0.05$ cyc/weight at $G \geq 64$ |
| 5.3 | Never recompute | $\beta^\star \approx 8$ MB/s; fetch wins by $355\times$ |

# Appendix B — Extreme-Value Reference

$\mathbb{E}[\max]$ of $n$ standard normals:

| $n$ | 10 | 20 | 30 | 32 | 50 | 64 | 100 | 128 | 200 | 256 | 500 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| $\mathbb{E}[\max]$ | 1.539 | 1.867 | 2.043 | 2.07 | 2.249 | 2.32 | 2.508 | 2.58 | 2.746 | 2.83 | 3.037 |

# Appendix C — Deck Corrections

| Slide 7 as written | Corrected |
|---|---|
| "$12\times$ more feed rate" | "$12\times$ weight-stream ceiling at full residency; $4$–$7\times$ end-to-end (Cor. 4.4)" |
| Path B $1.939$ MiB $\leq$ L2 | Feasible only at $\eta \geq 0.97$; re-tile to $\approx 1.2$ MiB (Prop. 3.1) |
| "$100\%$ behavior change" | Replace with the $S(h)$ curve — the nonlinearity *is* the story |
| Metadata amortization as the novelty | Reframe as cache-residency-driven granularity control; cite K-quants as prior art |
| Lloyd–Max in the kv hot path | Demote; lead with outlier extraction (Prop. 2.10 vs. Thm. 2.9) |
| "$\sim$90 ns" | Correct as drawn (DRAM edge) — label the edge explicitly so it is not read as L2 |

# Appendix D — References

Bennett, W.R. (1948) *Spectra of quantized signals.* BSTJ.
Panter, P. & Dite, W. (1951) *Quantization distortion in PCM.* Proc. IRE.
Lloyd, S. (1957/1982) *Least squares quantization in PCM.* IEEE Trans. Inf. Theory.
Max, J. (1960) *Quantizing for minimum distortion.* IRE Trans. Inf. Theory.
Goto, K. & van de Geijn, R. (2008) *Anatomy of high-performance matrix multiplication.* ACM TOMS.
Williams, S., Waterman, A., Patterson, D. (2009) *Roofline.* CACM.
Banner, R. et al. (2019) *Post-training 4-bit quantization (ACIQ).* NeurIPS.
Frantar, E. et al. (2023) *GPTQ.* ICLR.
Dettmers, T. et al. (2023) *QLoRA.* NeurIPS. — double quantization
Lin, J. et al. (2024) *AWQ.* MLSys.
Kim, S. et al. (2024) *SqueezeLLM.* ICML. — sensitivity-weighted k-means
Liu, Z. et al. (2024) *KIVI.* ICML. — 2-bit KV
Hooper, C. et al. (2024) *KVQuant.* NeurIPS.
Na, S. et al. (2024) *LLM inference on CPUs.* IISWC.
Zheng, L. et al. (2024) *SGLang / RadixAttention.* NeurIPS.
llama.cpp K-quant specification, GGUF format documentation (2023–).

---

*Prepared as an internal technical document. Every numeric claim above is either derived in place or reducible to one of the six experiments in Part VI. Claims that cannot survive E4 should not appear in an investor deck.*
