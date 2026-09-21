# PNTC Prototype Evidence Trace: Mechanistic Explanation & Score Decomposition

**PNTC Prototype Evidence Trace** makes the core cross-modal manifold topology mechanism directly transparent, verifiable, and explainable.

---

## 1. Research Intuition: What PNTC Detects

In nominal (defect-free) regions, RGB texture and 3D surface geometry correspond to the **same physical entity** on the normal industrial manifold. As a result:
- RGB features retrieve normal prototypes from a localized neighborhood $\mathcal{I}_{\text{rgb}}(p)$.
- XYZ features retrieve normal prototypes from the same neighborhood $\mathcal{I}_{\text{xyz}}(p)$.
- Prototype overlap is high ($4/5$ or $5/5$ shared IDs), and Jensen-Shannon divergence is near zero ($D_{\text{JS}} \approx 0.10$).

In anomalous regions, subtle defects cause one or both modalities to map into contradictory, distant normal prototype clusters:
- RGB may retrieve a patch resembling a dark seam.
- XYZ may retrieve a patch resembling a smooth flat surface.
- Their retrieved neighborhoods are **disjoint** (overlap $0/5$, Jaccard $0.00$), and normalized JS divergence approaches $1.00$.

---

## 2. Mathematical Score Decomposition

The total PNTC anomaly score at test patch $p$ is decomposed into:
$$A_{\text{PNTC}}(p) = A_{\text{base}}(p) + \lambda \cdot G(p) \cdot T(p)$$

Where:
- $A_{\text{base}}(p) = \frac{1}{2}\tilde{A}_{\text{rgb}}(p) + \frac{1}{2}\tilde{A}_{\text{xyz}}(p)$: Baseline decision fusion score.
- $T(p) = \frac{1}{\ln 2} \text{JSD}(P_{\text{rgb}} \,\|\, P_{\text{xyz}}) \in [0, 1]$: Normalized Jensen-Shannon divergence over union prototype set $\mathcal{U} = \mathcal{I}_{\text{rgb}} \cup \mathcal{I}_{\text{xyz}}$.
- $G(p) \in [0, 1]$: Normal retrieval confidence gate derived strictly from train-normal dispersion statistics.
- $\lambda = 0.35$: Fixed topology weighting constant.

### Exact Bit-for-Bit Verification:
The inspection suite verifies that the decomposed components reconstruct the stored patch score:
$$A_{\text{base}}(p) + \lambda \cdot G(p) \cdot T(p) \equiv A_{\text{PNTC}}(p) \quad (\pm 10^{-4})$$

---

## 3. Prototype Evidence Table Example

```
RGB TOP-5 RETRIEVAL
Rank  Prototype ID  Distance  Source Sample              Spatial Patch
1     #124          0.181     train/cookie/good/012      (20, 24)
2     #381          0.217     train/cookie/good/005      (18, 22)
3     #742          0.238     train/cookie/good/019      (21, 25)
4     #991          0.267     train/cookie/good/002      (19, 23)
5     #205          0.291     train/cookie/good/028      (20, 26)

XYZ TOP-5 RETRIEVAL
Rank  Prototype ID  Distance  Source Sample              Spatial Patch
1     #807          0.163     train/cookie/good/008      (38, 14)
2     #912          0.209     train/cookie/good/014      (39, 15)
3     #441          0.231     train/cookie/good/021      (37, 12)
4     #635          0.258     train/cookie/good/003      (38, 16)
5     #101          0.279     train/cookie/good/027      (36, 13)

TOPOLOGY METRICS:
  Shared Prototype IDs : 0 / 5
  Jaccard Overlap      : 0.0000
  Reciprocal Rank Sum  : 0.0000
  JS Divergence T(p)   : 0.9124
  Confidence Gate G(p) : 0.8800
  Topology Addition    : +0.2810
  Base Anomaly Score   : 0.6200
  Final PNTC Score     : 0.9010
```

---

## 4. Deterministic "Why Flagged" Explanation Generator

The system deterministic generates an evidence-grounded explanation:
> *"RGB and XYZ individually retrieve plausible normal samples, but they map the same physical test region to largely different normal prototype neighborhoods. Their top-5 overlap is 0/5 (disjoint normal neighborhoods) and normalized JS divergence is very high (0.91). Under a high confidence gate (0.88), this topological inconsistency contributes +0.281 additional anomaly evidence through PNTC."*

No remote LLM or probabilistic guesswork is involved.
