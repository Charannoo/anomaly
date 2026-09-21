# PNTC Inspect — Web Application Design System

## 1. Design Philosophy: "Precision Industrial Instrumentation"

**PNTC Inspect** is designed as a research-grade, production-ready industrial metrology interface. It takes visual cues from high-precision scientific equipment (Keyence, ZEISS, Hexagon, Cognex) and modern developer tooling (Linear, Vercel), prioritizing:
- **Accuracy over spectacle**: No AI clichés, no cartoon icons, no giant neon gradients, no glassmorphism.
- **High information density**: Clean, structured, tabular alignment with tabular numerals for micro-measurements.
- **Intentional, quiet hierarchy**: Thin 1px borders, subtle surface elevations, disciplined 8px spacing, and restrained micro-motion (150–220ms).

---

## 2. Color Tokens

### Dark Mode (Primary Palette)
```css
:root {
  /* Backgrounds */
  --bg-app: #0B0D10;         /* Primary canvas */
  --bg-subtle: #101318;      /* Secondary canvas / sidebar */
  --bg-panel: #15191F;       /* Cards & inspection panels */
  --bg-surface: #191E25;     /* Raised elements, inputs, hover states */
  --bg-active: #212832;      /* Selected table rows, active tabs */

  /* Borders */
  --border-default: rgba(255, 255, 255, 0.08);
  --border-subtle: rgba(255, 255, 255, 0.05);
  --border-emphasized: rgba(255, 255, 255, 0.14);
  --border-focus: #5BB8C4;

  /* Typography */
  --text-primary: #F3F5F7;   /* Headers, primary numbers, titles */
  --text-secondary: #A7AFBA; /* Labels, table headers, descriptions */
  --text-muted: #6F7884;     /* Secondary metadata, units, hints */
  --text-disabled: #424852;

  /* Functional Industrial Accents */
  --accent-primary: #5BB8C4; /* Precision cyan-teal accent */
  --accent-hover: #71C7D1;
  --accent-subtle: rgba(91, 184, 196, 0.12);

  /* Status Colors */
  --status-normal: #55B98A;  /* Controlled nominal state */
  --status-normal-bg: rgba(85, 185, 138, 0.12);
  --status-anomaly: #E96B6B; /* Defect detected */
  --status-anomaly-bg: rgba(233, 107, 107, 0.12);
  --status-review: #D4A95B;  /* Review guard / uncertainty */
  --status-review-bg: rgba(212, 169, 91, 0.12);
  --status-info: #6C96D8;    /* General system info */
  --status-info-bg: rgba(108, 150, 216, 0.12);
}
```

---

## 3. Typography Hierarchy

- **Font Family**: Inter (or Geist), with `font-variant-numeric: tabular-nums` enforced across all measurement readouts and tables.
- **Monospace**: `JetBrains Mono` or system mono, restricted exclusively to prototype IDs, coordinate tuples, git hashes, and model IDs.

| Role | Font Size | Line Height | Weight | Letter Spacing | Case |
|---|---|---|---|---|---|
| **Page Title** | 22–24px | 28px | 600 (SemiBold) | -0.02em | Normal |
| **Section Title** | 16–18px | 24px | 600 (SemiBold) | -0.01em | Normal |
| **Card / Panel Header**| 13–14px | 18px | 600 (SemiBold) | 0.02em | Uppercase / Normal |
| **Body Text** | 14px | 20px | 400 (Regular) | 0 | Normal |
| **Dense Table Cell** | 13px | 18px | 400–500 | 0 | Normal |
| **Metric Readout** | 20–28px | 32px | 600 (SemiBold) | -0.02em | Tabular |
| **Metadata / Unit** | 11–12px | 16px | 400–500 | 0.02em | Normal / Mono |

---

## 4. Spacing & Grid System

Strict **8px geometric spacing scale**:
- `4px`: Micro-gaps between badges and text, inline tags.
- `8px`: Dense inner-padding, input vertical padding, icon-text gap.
- `12px`: Standard button padding, table row vertical padding.
- `16px`: Card padding, section gaps.
- `24px`: Main container padding, grid column gaps.
- `32px`: Major layout split margins.
- `48px`: Page header margins.

---

## 5. Border Radius & Elevation

- **Buttons & Inputs**: `6px`
- **Cards, Modals & Inspection Panels**: `8px`
- **Status Indicator Badges**: `4px`
- **Shadows**: Extremely restrained. Avoid large diffusion; use 1px border contrast with surface elevation:
  - Surface 0 (`#0B0D10`): Canvas
  - Surface 1 (`#101318`): Sidebar, toolbars
  - Surface 2 (`#15191F`): Active workspace cards, panels
  - Surface 3 (`#191E25`): Hover states, inputs, dropdown menus

---

## 6. Interaction Rules & Micro-Motion

- **Transition Durations**: `150ms` (standard interactions, hover, color shifts) to `200ms` (panel collapses, modals).
- **Easing**: `cubic-bezier(0.16, 1, 0.3, 1)` (ease-out crisp).
- **Prohibited**: Bouncing animations, floating decorative cards, neon glowing pulses, auto-rotating hero models.
- **Defect Selection**: Clicking a defect in the 2D overlay highlights the defect row in the inspector with a subtle left border highlight (`#5BB8C4`) and updates all volumetric readouts.
