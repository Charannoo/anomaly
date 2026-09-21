"""PNTC Review Guard: Multi-tiered uncertainty quantification and deployment manual-review guard."""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from .schema import ReviewGuardInfo, QualityInfo, VolumeInfo


def evaluate_review_guard(
    image_score: float,
    threshold: float,
    quality: QualityInfo,
    volume: Optional[VolumeInfo] = None,
    is_physical_calibrated: bool = True,
    train_normal_score_mean: float = 0.20,
    train_normal_score_std: float = 0.08
) -> ReviewGuardInfo:
    """Evaluate decision certainty, measurement reliability, and manual-review triggers.
    
    CRITICAL RULE:
        Does NOT modify the canonical PNTC benchmark decision (normal/anomalous).
        Adds an operational deployment recommendation:
            ACCEPT_NORMAL | DEFECT_DETECTED | MANUAL_REVIEW_RECOMMENDED
            
    Args:
        image_score: Continuous anomaly score from PNTC.
        threshold: Operating decision threshold.
        quality: QualityInfo instance.
        volume: Optional VolumeInfo instance.
        is_physical_calibrated: Boolean indicating verified physical coordinates.
        train_normal_score_mean: Training normal mean anomaly score.
        train_normal_score_std: Training normal anomaly score standard deviation.
        
    Returns:
        ReviewGuardInfo instance.
    """
    # 1. Canonical PNTC benchmark decision (NEVER altered)
    pntc_decision = "anomalous" if image_score >= threshold else "normal"

    # 2. Tier 1: Detection Certainty (Margin from decision threshold)
    margin = float(abs(image_score - threshold))
    # Normalized margin relative to train-normal score variance
    margin_z = margin / max(train_normal_score_std, 1e-4)

    if margin_z >= 2.0:
        certainty = "High"
    elif margin_z >= 0.8:
        certainty = "Moderate"
    else:
        certainty = "Borderline"

    # 3. Tier 2: Geometry Measurement Reliability
    surface_fit_conf = quality.surface_fit_confidence
    coverage = quality.xyz_valid_fraction
    
    if coverage >= 0.80 and surface_fit_conf >= 0.75 and is_physical_calibrated:
        measurement_reliability = "High"
    elif coverage >= 0.50 and surface_fit_conf >= 0.50:
        measurement_reliability = "Moderate"
    else:
        measurement_reliability = "Low"

    # 4. Trigger Analysis for Manual Review
    reasons: List[str] = []

    # Sensor quality triggers
    if quality.xyz_valid_fraction < 0.60:
        reasons.append(f"LOW_XYZ_COVERAGE: Only {quality.xyz_valid_fraction*100:.1f}% valid 3D points inside defect region")

    if quality.rgb_contrast < 0.08:
        reasons.append("LOW_RGB_QUALITY: Low visual contrast / potential illumination saturation")

    # Boundary proximity triggers
    if quality.object_boundary_proximity > 0.75:
        reasons.append(f"OBJECT_BOUNDARY_REGION: Defect lies in close proximity to object silhouette edge ({quality.object_boundary_proximity:.2f})")

    # Surface fit triggers
    if quality.surface_fit_confidence < 0.60:
        reasons.append(f"LOW_SURFACE_FIT_CONFIDENCE: Reference surface fit residual is elevated (confidence: {quality.surface_fit_confidence:.2f})")

    # Retrieval stability triggers
    if quality.retrieval_stability == "unstable":
        reasons.append("UNSTABLE_NEIGHBOR_RETRIEVAL: Narrow margin gap between top-k prototype neighbors")

    # Decision boundary proximity trigger
    if certainty == "Borderline":
        reasons.append(f"NEAR_DECISION_THRESHOLD: PNTC score ({image_score:.3f}) lies within {margin:.3f} of operating threshold ({threshold:.3f})")

    # Region size trigger
    if 0 < quality.region_size_px < 15:
        reasons.append(f"SMALL_REGION: Defect area is small ({quality.region_size_px} px), prone to noise fluctuations")

    # Physical calibration trigger
    if not is_physical_calibrated:
        reasons.append("MISSING_PHYSICAL_CALIBRATION: Physical coordinate scaling is uncalibrated; volumetric claims are native-only")

    # 5. Operational Inspection Status Determination
    # Severe disqualifying flags for automated defect acceptance:
    severe_review_triggers = [
        "LOW_XYZ_COVERAGE" in r and quality.xyz_valid_fraction < 0.50 for r in reasons
    ] + [
        "NEAR_DECISION_THRESHOLD" in r for r in reasons
    ] + [
        "LOW_SURFACE_FIT_CONFIDENCE" in r and quality.surface_fit_confidence < 0.40 for r in reasons
    ]

    if pntc_decision == "anomalous":
        if any(severe_review_triggers) or len(reasons) >= 3:
            inspection_status = "MANUAL_REVIEW_RECOMMENDED"
            recommendation_text = (
                f"MANUAL INSPECTION RECOMMENDED. PNTC detected an abnormal region (score: {image_score:.3f} vs "
                f"threshold: {threshold:.3f}), but {len(reasons)} inspection warnings were triggered. "
                f"The canonical anomaly detection decision is preserved, but geometric/depth estimates "
                f"require manual validation."
            )
        else:
            inspection_status = "DEFECT_DETECTED"
            recommendation_text = (
                f"DEFECT DETECTED. PNTC anomaly evidence is strong with {certainty.lower()} certainty "
                f"and {measurement_reliability.lower()} measurement reliability."
            )
    else:
        if certainty == "Borderline":
            inspection_status = "MANUAL_REVIEW_RECOMMENDED"
            recommendation_text = (
                f"MANUAL INSPECTION RECOMMENDED. Sample is below operating threshold ({image_score:.3f} < {threshold:.3f}), "
                f"but score is near the boundary; visual spot-check recommended."
            )
        else:
            inspection_status = "ACCEPT_NORMAL"
            recommendation_text = (
                f"ACCEPT NOMINAL. Sample conforms to normal prototype manifold with {certainty.lower()} certainty."
            )

    return ReviewGuardInfo(
        pntc_decision=pntc_decision,
        inspection_status=inspection_status,
        decision_certainty=certainty,
        decision_margin=round(margin, 4),
        measurement_reliability=measurement_reliability,
        reasons=reasons,
        recommendation_text=recommendation_text
    )
