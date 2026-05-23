# Font Labeling Protocol

This document defines how every font in `fonts.csv` is assigned to exactly one of four style classes. Apply it consistently. When in doubt, prefer the class that captures the font's **primary visual signature** — the feature a designer would name first when describing the font to a colleague.

## The four classes

### 1. `serif`
A font with **terminal serifs** — small projecting strokes at the ends of letterforms (the "feet" on a lowercase 'n', the "heads" on a lowercase 'h').

Includes: old-style, transitional, modern, and slab serifs.

Tests:
- Does the lowercase 'n' have visible serifs at the bottom of its two strokes? → `serif`
- Does the lowercase 'a' have a thin spur at the bottom-right? → `serif`

Examples: Lora, EB Garamond, Playfair Display, Cormorant, Crimson Text, Times New Roman, Georgia, PT Serif.

### 2. `sans-serif`
A font **without serifs**, with relatively **uniform stroke width** (low stress).

Tests:
- Does the lowercase 'n' end in a clean vertical cut at the bottom? → likely `sans-serif`
- Is the stroke width nearly constant around the lowercase 'o'? → likely `sans-serif`

Examples: Inter, Roboto, Open Sans, Work Sans, Nunito, Helvetica, Arial, Lato.

### 3. `calligraphic` (also called "script" or "handwriting")
A font **explicitly modeled on pen-written or brush-written letterforms**. Strokes show clear pen stress (thick-to-thin transitions) and/or connected/flowing letterforms.

Tests:
- Do the lowercase letters appear to connect (or attempt to connect) as if written in cursive? → `calligraphic`
- Do the strokes have dramatic thick-to-thin variation consistent with a pen nib or brush? → `calligraphic`
- Does the font visibly mimic handwriting (hand-drawn appearance, irregular baseline)? → `calligraphic`

Examples: Great Vibes, Dancing Script, Sacramento, Pinyon Script, Allura, Pacifico (borderline; see note).

### 4. `display` (also called "decorative")
A font designed primarily for **headlines or display use**, with **strong stylistic personality** that would be hard to read in long body text. Often includes unusual proportions, heavy weights, geometric shapes, or stylized features that don't fit the other three categories.

Tests:
- Would this font be uncomfortable to read in a 1,000-word essay? → likely `display`
- Does it have unusual or exaggerated proportions (very tall, very wide, very heavy)? → `display`
- Is it a serif or sans-serif but with strong decorative twists (e.g., layered, outlined, hatched)? → `display`

Examples: Lobster (heavy script-display hybrid; we label as `display`), Bungee, Abril Fatface, Monoton, Bowlby One.

## Resolving ambiguous cases

When a font sits between two classes, decide by **primary signature**:

- **Display vs. calligraphic.** If the font has decorative flourishes but is fundamentally pen-written → `calligraphic`. If it has script-like elements but is fundamentally a stylized headline face → `display`. Example: Lobster has flowing connections (calligraphic-ish) but heavy weight and dramatic personality for headlines (display-ish). The community uses it as a display font, so we label it `display`.
- **Serif vs. display.** If a serif font is fundamentally readable in body text → `serif`. If the serif is exaggerated for personality at large sizes → `display`. Example: Playfair Display has serifs but is built for display use — but the serif structure is more visually dominant than the display quality, so we label it `serif`.
- **Sans-serif vs. display.** Geometric monoline fonts that are simply unconventional → `display`. Clean sans-serifs that work in body text → `sans-serif`.

## The Google Fonts category as a starting point

Every font in the `google/fonts` repository has a `METADATA.pb` file with a `category:` field. The mapping is:

| Google category | Our class |
|---|---|
| `SERIF` | `serif` |
| `SANS_SERIF` | `sans-serif` |
| `HANDWRITING` | `calligraphic` |
| `DISPLAY` | `display` |
| `MONOSPACE` | (excluded — not one of our four classes) |

**Use Google's labels as a default, then audit.** Open the font in a font viewer or at fonts.google.com, render the alphabet, and apply the tests above. Override the Google label if your audit disagrees. Document any override in `fonts.csv` by writing your final label in the `style_class` column (the `google_category` column preserves the original).

## A practical workflow

1. Open `fonts.csv`.
2. For each row, look up the font on https://fonts.google.com/specimen/[font_name].
3. Read the lowercase alphabet at large size.
4. Apply the tests above. Confirm or override the label.
5. If you change a label, add a one-line comment in the `notes` column explaining why.

## Why this matters

The labeling is the only place in the entire pipeline where human judgment enters. Every downstream number (macro-F1, confusion matrix, feature importance) is conditional on these labels being right. If the labels are inconsistent, the results are too. The lit review explicitly flags this as a methodological choice that must be documented.
