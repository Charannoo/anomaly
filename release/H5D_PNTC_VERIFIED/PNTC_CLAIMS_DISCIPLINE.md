# Publication Claims Discipline & Scientific Guardrails

**Document**: `docs/PNTC_CLAIMS_DISCIPLINE.md`  
**Purpose**: Explicit binding guidelines on permissible and impermissible scientific claims in all papers, reports, and release documentation.

---

## 1. Strictly Permissible Benchmark Claims

1. **Declared Protocol Accuracy**:
   > *"PNTC achieves **0.96541 I-AUROC**, **0.99416 P-AUROC**, and **0.96939 AUPRO@0.3** under our declared evaluation protocol on MVTec 3D-AD."*

2. **Comparison with Reported M3DM**:
   > *"PNTC exceeds M3DM's reported values on all three metrics ($0.9654$ vs $0.9450$ I, $0.9942$ vs $0.9920$ P, $0.9694$ vs $0.9640$ AUPRO), subject to the documented protocol/sample-manifest difference."*

3. **Comparison with Reported CFM**:
   > *"PNTC improves image AUROC ($0.9654$ vs $0.9540$) and pixel AUROC ($0.9942$ vs $0.9930$) over CFM (CVPR 2024), while CFM reports slightly higher AUPRO@0.3 ($0.9710$ vs $0.9694$)."*

4. **Comparison with Reported G2SF**:
   > *"PNTC remains below G2SF's reported results ($0.9710$ I / $0.9970$ P / $0.9790$ AUPRO), positioning PNTC as an efficient, training-free topological alternative rather than an absolute numerical replacement for learned metric networks."*

5. **Statistical Significance over Baseline**:
   > *"PNTC demonstrates statistically significant improvements over the frozen H4B decision fusion baseline across image AUROC ($p = 0.0078$, Wilcoxon) and localization AUPRO ($p = 0.0020$, Wilcoxon). Multi-seed evaluation confirms prototype stability ($\sigma < 0.00003$)."*

---

## 2. Strictly Impermissible Claims (Forbidden)

1. **DO NOT claim "State of the Art (SOTA)", "state of the art", "best overall", or "uniformly surpasses prior methods"**:
   - *Reason*: G2SF reports higher overall metrics and CFM reports higher AUPRO.
2. **DO NOT claim "beats G2SF"**:
   - *Reason*: G2SF is numerically higher across all three metrics.
3. **DO NOT claim "novel first-ever rank consistency"**:
   - *Reason*: Cross-modal rank consistency exists in general multimodal retrieval. Novelty is strictly narrowed to:
     > *"Paired normal-prototype retrieval topology consistency for RGB–3D industrial anomaly detection."*
4. **DO NOT claim exact same-manifest reproduction of M3DM without disclosure**:
   - *Reason*: M3DM CVPR 2023 explicitly states 1,137 test samples, whereas our evaluation uses the complete 1,197-sample manifest.
