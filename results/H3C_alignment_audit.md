# H3C Spatial Concat Alignment Audit

## Executive Summary

- **Phenomenon**: During 10-category evaluation, `H3C_Concat_Fusion` achieved reasonable Image AUROC (`0.8622`), but exhibited catastrophic collapse in pixel-level localization:
  - **Pixel AUROC**: `0.5949` (vs `0.9872` on H1 RGB and `0.9582` on H2 XYZ)
  - **AUPRO @ 0.3**: `0.2167` (vs `0.9433` on H1 RGB and `0.8613` on H2 XYZ)
- **Root Cause Identified**: A spatial index permutation formula in the legacy concat utility (`cache_utils.align_rgb_descriptors`) performed flat modulo striding rather than 2D spatial grid tiling.
- **Audit Verification**: Numerical coordinate tracing showed a spatial error of `55,664.0` Euclidean units on a $56 \times 56$ image plane between legacy concatenated patches and ground truth coordinates, which drops to exactly `0.0` with true 2D spatial interpolation.

---

## 1. Technical Analysis of the Alignment Bug

### Spatial Dimensions
- **RGB Feature Map (DINO ViT-B/8)**: $28 \times 28 = 784$ spatial patches ($D=768$).
- **XYZ Feature Map (Point-MAE)**: $56 \times 56 = 3,136$ interpolated points ($D=1,152$).
- **Goal of Concat Fusion ($1,920\text{-D}$)**: Each spatial position $(R, C)$ on the $56 \times 56$ surface must pair the XYZ point $(R, C)$ with the corresponding RGB patch $(\lfloor R/2 \rfloor, \lfloor C/2 \rfloor)$.

### The Legacy Permutation Error
The legacy implementation used:
```python
perm = torch.arange(3136)
a = perm % 784
c = perm // 784
perm = (4 * a + c) % 784
x = x[perm]
```
Tracing the first 8 output coordinate locations on the $56 \times 56$ grid:
```
Index 0: (row 0, col 0)  -> mapped to (0, 0)   [Correct]
Index 1: (row 0, col 1)  -> mapped to (0, 4)   [Incorrect: jumped 4 columns!]
Index 2: (row 0, col 2)  -> mapped to (0, 8)   [Incorrect: jumped 8 columns!]
Index 3: (row 0, col 3)  -> mapped to (0, 12)  [Incorrect: jumped 12 columns!]
Index 4: (row 0, col 4)  -> mapped to (0, 16)  [Incorrect]
Index 5: (row 0, col 5)  -> mapped to (0, 20)  [Incorrect]
Index 6: (row 0, col 6)  -> mapped to (0, 24)  [Incorrect]
Index 7: (row 0, col 7)  -> mapped to (1, 0)   [Incorrect: jumped to next row!]
```
Instead of expanding patch $(0,0)$ to the $2 \times 2$ block `{(0,0), (0,1), (1,0), (1,1)}`, the formula strided across every 4th column of the entire image.

### Why Image AUROC Survived but Localization Collapsed
- **Image-Level Detection**: Image anomaly scoring computes the max/tail distance over the entire set of patches in the image. Since all patches were present (merely permuted), the global maximum distance remained sensitive to whole-image anomalies.
- **Pixel-Level Localization**: The anomaly map at pixel $(r, c)$ received the distance of a patch from $(r, c+4k)$, scattering anomaly heatmaps randomly across the surface.

---

## 2. Correct 2D Spatial Alignment Formulation

To preserve spatial correspondence for concatenation:
```python
def align_rgb_descriptors_correct(desc):
    # desc: (N*784, D)
    n = desc.shape[0] // 784
    D = desc.shape[1]
    # Reshape to 2D image plane: (N, 28, 28, D) -> (N, D, 28, 28)
    feat_2d = desc.view(n, 28, 28, D).permute(0, 3, 1, 2)
    # Perform exact 2x spatial nearest-neighbor tile expansion:
    feat_56 = torch.nn.functional.interpolate(feat_2d, size=(56, 56), mode="nearest")
    # Return to canonical flattened sequence: (N*3136, D)
    return feat_56.permute(0, 2, 3, 1).reshape(n * 3136, D)
```
- **Spatial Alignment Error with Correct Implementation**: **`0.0`**

---

## 3. Recommendation for H3 / H4 Pipeline

1. **Avoid Raw Concat Memory Banks**: Even with perfect spatial alignment, unweighted concatenation of $768\text{-D}$ RGB and $1,152\text{-D}$ XYZ descriptors suffers from modality magnitude imbalance and high dimensionality ($1,920\text{-D}$).
2. **Standardize on Decision-Level / Map-Level Fusion (`H3D`)**: Fusing calibrated RGB anomaly maps ($28 \times 28 \to 224 \times 224$) and XYZ anomaly maps ($56 \times 56 \to 224 \times 224$) provides strictly superior pixel localization ($0.9897\text{ P-AUROC}$) without high-dimensional concat noise.
