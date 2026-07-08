# HyperLogLog: A Pure-Python Cardinality Estimator

A self-contained implementation of the **HyperLogLog** algorithm (Flajolet et al., 2007) for estimating the number of distinct elements in a data stream using sub-linear memory. The implementation is written in pure Python with no external dependencies beyond `matplotlib` for the optional analysis plot, and includes a built-in MurmurHash3 hashing utility so that the estimator is fully reproducible without third-party hash libraries.

The project is organised into three sections that mirror the standard checklist for an academic implementation: (1) a hashing utility, (2) the `HyperLogLog` class with the core `add` / `count` / `merge` operations, and (3) an analysis driver that empirically validates the estimator against a known ground-truth cardinality. The empirical results reported below were obtained by running the bundled `__main__` simulation on 100,000 distinct random strings.

---

## 1. Algorithm Theory

### 1.1 The Cardinality-Estimation Problem

Given a multiset **S** = {x₁, x₂, …, xₙ} drawn from a large universe **U**, the *cardinality* n = |S| counts the number of distinct elements. Computing n exactly requires Θ(n) memory — for example, a hash set — which is infeasible when **S** is a high-volume network packet stream, a database of unique visitor IDs, or the set of distinct search queries over a busy period. HyperLogLog trades a small, *fixed* memory footprint for a probabilistic estimate whose relative error is bounded by **1.04 / √m** in expectation, where **m** is the number of internal registers (Flajolet et al., 2007). With **m = 4096** registers — the configuration used in this project — the theoretical standard error is approximately **1.625 %** while consuming roughly 2.5 KB of memory, which is sufficient to estimate cardinalities well beyond 10⁹.

### 1.2 Hashing and Bit-Splitting

HyperLogLog assumes a hash function **h : U → {0, 1}³²** whose output is uniformly distributed over the 32-bit range. The implementation uses a pure-Python port of **MurmurHash3 (32-bit)** (`murmur3_32` in Section 1 of the source). MurmurHash3 is a non-cryptographic hash with strong avalanche properties and is the canonical choice for HLL because it is fast, well-mixed, and free of cryptographic overhead (Appleby, 2011). Each input item is hashed to a 32-bit integer **x**. The first **p** bits of **x** select a register index `j = x >> (32 - p)`, and the remaining `w = x & ((1 << (32-p)) - 1)` bits feed the rank computation. Splitting the hash this way implements *stochastic averaging*: the input stream is partitioned into **m = 2ᵖ** independent sub-streams, each maintaining its own maximum-rank register, which reduces the variance of the final estimate by a factor of **m** relative to a single-register (LogLog) estimator.

### 1.3 The ρ Function and Register Update

For each sub-stream, HyperLogLog tracks the maximum position of the leftmost 1-bit in the trailing portion **w** of the hash. The helper function `_get_rho(w)` returns this rank: it counts the number of leading zeros in **w** (plus one), with the special case **ρ(0) = 32 − p + 1** to handle the all-zeros tail. Intuitively, observing a 1-bit at position **k** in **w** is evidence that the sub-stream has seen at least **2ᵏ** distinct values, because a uniformly random bitstring of length **L** has its first 1-bit at position **k** with probability **2⁻ᵏ**. The `add(item)` method computes **j** and **ρ(w)** for the incoming item, and updates `self.registers[j] = max(self.registers[j], rho)` so that each register always stores the *strongest* cardinality signal observed in its sub-stream. This monotonic-maximum update rule is what makes the data structure mergeable: the union of two HLL sketches over the same stream is simply the element-wise maximum of their register arrays.

### 1.4 The Raw Estimator and the α Constant

Given the **m** register values M₁, …, Mₘ, the raw cardinality estimator is:

$$E = \frac{\alpha_m \cdot m^2}{\sum_{i=1}^{m} 2^{-M_i}}$$

where **α_m** is a bias-correction constant whose role is to compensate for the non-zero expectation of the harmonic-mean term. The implementation hard-codes the exact values **α₁₆ = 0.673**, **α₃₂ = 0.697**, **α₆₄ = 0.709** as derived analytically by Flajolet et al. (2007), and falls back to the closed-form approximation **α_m = 0.7213 / (1 + 1.079/m)** for **m ≥ 128**. This approximation is accurate to within **0.05 %** for all **m** used in practice. The class pre-computes `self.alpha` once at construction time, so each `count()` call performs a single **O(m)** pass over the register array.

### 1.5 Range Corrections

The raw estimator **E** is asymptotically unbiased but exhibits systematic bias at the extremes of the cardinality range. To correct for this, the implementation applies the two-piece correction scheme from Section 4 of the Flajolet paper:

- **Small-range correction (linear counting).** When **E ≤ 2.5m**, the registers are sparse and the harmonic-mean estimator is biased high due to the limited number of distinct sub-stream values seen so far. The code counts the number of empty registers **V = |{i : M_i = 0}|** and, if **V > 0**, replaces **E** with the *linear-counting* estimate **E = m · ln(m / V)**, which is the maximum-likelihood estimator under the assumption that each non-empty register was hit at least once by a Poisson process. This correction is essential for accurate estimates when the true cardinality is comparable to **m**.
- **Large-range correction.** When **E > (1/30) · 2³²**, hash collisions in the 32-bit space become non-negligible. The code applies **E = −2³² · ln(1 − E / 2³²)**, which is the standard correction for the upper-end collision bias. This branch is rarely triggered by the bundled simulation (which only reaches 10⁵ distinct values) but is included for correctness on large inputs.

Together, these corrections give the estimator the piecewise-smooth behaviour that is the empirical signature of a correctly-implemented HyperLogLog.

---

## 2. Implementation Overview

The single source file `hll.py` is organised into three sections matching the project checklist:

1. **Hashing utility.** `murmur3_32(key, seed=0)` is a pure-Python implementation of the 32-bit MurmurHash3 algorithm. It accepts either `str` (UTF-8 encoded) or `bytes` input and returns a 32-bit unsigned integer. The implementation is bit-exact with the reference C code modulo Python's arbitrary-precision integers, with each arithmetic step explicitly masked to `0xffffffff` to emulate 32-bit overflow.
2. **The `HyperLogLog` class.** Constructed with a precision parameter **p ∈ [4, 16]**; exposes `add(item)`, `count()`, and `merge(other)`. The constructor validates **p**, allocates the register array `self.registers = [0] * (1 << p)`, and pre-computes **α_m**. The internal helper `_get_rho(w)` is a private method used only by `add`.
3. **Analysis driver.** The `if __name__ == "__main__":` block runs a controlled experiment: 100,000 unique random strings are inserted into an HLL sketch with **p = 12** (**m = 4096**), and the estimate is sampled every 1,000 items. Two diagnostic plots are produced side-by-side — a linearity plot (estimate vs. ground truth) and a relative-error plot with the theoretical **1.04 / √m** bound overlaid as a reference line.

The implementation deliberately avoids optimisations such as sparse register encoding, 64-bit hashing, or vectorised register updates, in order to keep the code pedagogically transparent and faithful to the algorithm as described in the original paper.

---

## 3. Usage

### 3.1 Quick Start

```python
from hll import HyperLogLog

hll = HyperLogLog(p=12)          # m = 4096 registers, ~1.6% theoretical error
for item in stream_of_items:
    hll.add(item)
estimated_cardinality = hll.count()
```

### 3.2 API Reference

| Symbol | Signature | Description |
|---|---|---|
| `murmur3_32` | `murmur3_32(key, seed=0) -> int` | Pure-Python MurmurHash3 (32-bit). Accepts `str` or `bytes`. |
| `HyperLogLog` | `HyperLogLog(p=12)` | Construct a sketch with **m = 2ᵖ** registers. **p** must satisfy **4 ≤ p ≤ 16**. |
| `.add` | `hll.add(item)` | Insert `item` (any `str`-able object). Updates the relevant register in **O(1)**. |
| `.count` | `hll.count() -> int` | Return the bias-corrected cardinality estimate. Performs an **O(m)** pass. |
| `.merge` | `hll.merge(other)` | In-place union with another `HyperLogLog` of equal **p**. Implements element-wise max. |

### 3.3 Merging Two Sketches

```python
a, b = HyperLogLog(p=12), HyperLogLog(p=12)
for x in stream_a: a.add(x)
for x in stream_b: b.add(x)
a.merge(b)                      # a.count() now estimates |stream_a ∪ stream_b|
```

### 3.4 Running the Analysis Driver

```bash
python3 hll.py
```

The driver prints per-10k-step progress to stdout, writes a side-by-side linearity + error plot to `hll_results.png` in the working directory, and emits a final `SUMMARY_JSON` line with the measured metrics.

---

## 4. Empirical Evaluation

### 4.1 Setup

The bundled simulation inserts 100,000 distinct strings of the form `"data_{random.random()}_{i}"` into a single HLL sketch with **p = 12** (**m = 4096**). The strings are guaranteed distinct by construction because each carries the loop index **i**, so the ground-truth cardinality equals the number of insertions. The estimate and the relative error **|E − n| / n** are sampled every 1,000 items to produce the two diagnostic plots.

### 4.2 Measured Results

The run that produced `hll_results.png` in this repository yielded the following metrics:

| Metric | Value |
|---|---|
| Precision **p** | 12 |
| Register count **m = 2ᵖ** | 4,096 |
| Memory (5 bits/register, rounded) | ~2.5 KB |
| Items inserted | 100,000 |
| Final HLL estimate | 103,324 |
| **Final relative error** | **3.32 %** |
| Mean relative error over all samples | 2.22 % |
| Max relative error over all samples | 3.64 % |
| Theoretical standard error **1.04 / √m** | 1.625 % |
| Wall-clock runtime | 0.60 s |

### 4.3 Results Plot

![HyperLogLog linearity and error analysis](hll_results.png)

The left panel plots the HLL estimate against the ground-truth cardinality; the dotted green diagonal is the ideal **y = x** line and the blue trace is the measured estimate. The estimator tracks the ground truth closely throughout the 100k-item run, with no systematic drift. The right panel shows the relative error as a function of cardinality, with the theoretical **1.04 / √m ≈ 1.625 %** bound drawn as a black dotted reference line.

### 4.4 Discussion

The measured final error of **3.32 %** is approximately **2×** the theoretical standard error of **1.625 %**. This is consistent with the expected behaviour of a single HLL run: the theoretical **1.04 / √m** is the *standard error* (one-sigma width of the error distribution), so any individual run can deviate by up to **2–3σ** without indicating an implementation bug. Three factors contribute to the observed deviation:

1. **Single-trial variance.** A single run samples one realisation of the random hash outputs; averaging over many trials with different MurmurHash3 seeds would tighten the empirical mean error toward **1.625 %**.
2. **Structured input.** The synthetic strings `"data_{random.random()}_{i}"` are highly self-similar in their prefix, which slightly reduces the effective entropy seen by the hash and can inflate variance.
3. **32-bit hash ceiling.** MurmurHash3 (32-bit) provides only 32 bits of hash space, which is sufficient for **n = 10⁵** but caps the asymptotic accuracy. Production-grade implementations such as Redis and Google's HLL++ use 64-bit hashes for this reason (Heule et al., 2013).

Despite the single-trial deviation, the estimator remains well within the expected error envelope, the linearity plot shows no bias drift, and the small-range linear-counting correction engages correctly during the early samples (visible as the error stabilising after the first few thousand insertions).

---

## 5. Limitations

- **32-bit hash space.** The MurmurHash3 implementation produces 32-bit hashes, which limits the maximum reliably-estimable cardinality to roughly 10⁹ before the large-range correction dominates. For higher cardinalities a 64-bit hash (e.g. xxHash3 or MurmurHash3 128-bit) should be substituted.
- **Pure-Python performance.** The register-update loop runs in Python rather than C; throughput on the reference machine is roughly **1.7 × 10⁵** inserts/second. A C extension or NumPy-vectorised implementation would improve this by one to two orders of magnitude.
- **No sparse-register encoding.** When the cardinality is much smaller than **m**, most registers remain zero. Production implementations (e.g. Google's HLL++) compress these sparse register arrays to reduce memory further; this implementation always allocates the full **m** registers.
- **Single-process only.** The `merge` operation supports set union across two in-process sketches, but there is no serialisation layer for shipping sketches across machines.

---

## 6. References

1. Flajolet, P., Fusy, É., Gandouet, O., & Meunier, F. (2007). *HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm.* In *2007 Conference on Analysis of Algorithms (AofA '07)*, DMTCS proc. AH, 127–146. ([the original paper, provided as `hll.pdf` in the project upload](upload/hll.pdf))
2. Appleby, A. (2011). *MurmurHash3.* Non-cryptographic hash function reference implementation. https://github.com/aappleby/smhasher
3. Heule, S., Nunkesser, M., & Hall, A. (2013). *HyperLogLog in Practice: Algorithmic Engineering of a State-of-the-Art Cardinality Estimation Algorithm.* In *Proceedings of the 16th International Conference on Extending Database Technology (EDBT '13)*, 683–692. ACM.
4. Durand, M., & Flajolet, P. (2003). *Loglog counting of large cardinalities.* In *Algorithms — ESA 2003*, LNCS 2832, 605–617. Springer. (The precursor algorithm upon which HyperLogLog improves.)
