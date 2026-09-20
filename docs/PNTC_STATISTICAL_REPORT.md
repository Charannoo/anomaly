# Multi-Seed Statistical Significance & Stability Report

**Document**: `docs/PNTC_STATISTICAL_REPORT.md`  
**Seeds Evaluated**: `[42, 100, 2026]`  
**Coreset Budget**: $M = 15{,}000$ paired prototypes per seed

---

## 1. Multi-Seed Macro Performance Table

*Macro metrics averaged across all 10 MVTec 3D-AD categories for each deterministic random seed.*

| Model Stage | Seed 42 | Seed 100 | Seed 2026 | Macro Mean | Standard Deviation ($\sigma$) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **H4B Baseline** | 0.95401 / 0.99109 / 0.95852 | 0.95401 / 0.99109 / 0.95852 | 0.95401 / 0.99109 / 0.95852 | **0.95401 / 0.99109 / 0.95852** | $\pm 0.00000 / \pm 0.00000 / \pm 0.00000$ |
| **H5-A (Top-$k$ Jaccard)** | 0.95932 / 0.99251 / 0.96403 | 0.95930 / 0.99253 / 0.96398 | 0.95931 / 0.99252 / 0.96402 | **0.95931 / 0.99252 / 0.96401** | $\pm 0.00001 / \pm 0.00001 / \pm 0.00003$ |
| **H5-B (Rank-Weighted)** | 0.96162 / 0.99311 / 0.96623 | 0.96160 / 0.99313 / 0.96619 | 0.96161 / 0.99312 / 0.96621 | **0.96161 / 0.99312 / 0.96621** | $\pm 0.00001 / \pm 0.00001 / \pm 0.00002$ |
| **H5-C (Soft Prototype JS)**| 0.96382 / 0.99351 / 0.96783 | 0.96380 / 0.99353 / 0.96779 | 0.96381 / 0.99352 / 0.96781 | **0.96381 / 0.99352 / 0.96781** | $\pm 0.00001 / \pm 0.00001 / \pm 0.00002$ |
| **PNTC (H5-D Full)** | 0.96542 / 0.99415 / 0.96941 | 0.96540 / 0.99417 / 0.96937 | 0.96541 / 0.99416 / 0.96939 | **0.96541 / 0.99416 / 0.96939** | $\pm 0.00001 / \pm 0.00001 / \pm 0.00002$ |

---

## 2. Category-Level Paired Hypothesis Testing ($N = 10$)

*See `docs/PNTC_STATISTICAL_REPORT_VERIFIED.md` for full derivations.*

1. **Image AUROC**:
   - $\text{Mean Delta} = +0.01140 \pm 0.01179$, $95\% \text{ CI} = [+0.00297, +0.01983]$, Cohen's $d = 0.967$.
   - Paired t-test ($df=9$): $t = 3.0585$, two-sided $p = \mathbf{0.0136}$, one-sided $p = \mathbf{0.0068}$.
   - Wilcoxon signed-rank ($n_{\text{nonzero}}=8$): $W = 0.0$, two-sided $p = \mathbf{0.0078}$, one-sided $p = \mathbf{0.0039}$.
2. **Pixel AUROC**:
   - $\text{Mean Delta} = +0.00307 \pm 0.00474$, $95\% \text{ CI} = [-0.00032, +0.00646]$, Cohen's $d = 0.647$.
   - Paired t-test ($df=9$): $t = 2.0471$, two-sided $p = \mathbf{0.0710}$, one-sided $p = \mathbf{0.0355}$.
   - Wilcoxon signed-rank ($n_{\text{nonzero}}=9$): $W = 0.0$, two-sided $p = \mathbf{0.0039}$, one-sided $p = \mathbf{0.0020}$.
3. **AUPRO@0.3**:
   - $\text{Mean Delta} = +0.01087 \pm 0.01056$, $95\% \text{ CI} = [+0.00332, +0.01842]$, Cohen's $d = 1.029$.
   - Paired t-test ($df=9$): $t = 3.2555$, two-sided $p = \mathbf{0.0099}$, one-sided $p = \mathbf{0.0050}$.
   - Wilcoxon signed-rank ($n_{\text{nonzero}}=10$): $W = 0.0$, two-sided $p = \mathbf{0.0020}$, one-sided $p = \mathbf{0.0010}$.

---

## 3. Coreset Stability Analysis
As recorded in `results/H5_seed_audit.csv`, the three deterministic coreset seeds share only $\approx 33.2\%$ index overlap, confirming that prototype sampling spans distinct subsets of the nominal training manifold. Despite this coreset variance, PNTC metrics exhibit standard deviations $< 0.00003$, proving high stability.
