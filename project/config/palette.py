"""One palette for every figure: Okabe-Ito, colour-vision-deficiency safe.

Colour is assigned by the JOB it does, hues taken in a FIXED SLOT ORDER -- never cycled, never
generated. A 7th category folds into NEUTRAL rather than getting an invented hue.

SIX SLOTS IS A MEASURED CEILING, checked all-pairs on a white surface:
  Okabe-Ito 6, this order   PASS -- worst dE 7.6 CVD (floor band), 15.6 normal
  Okabe-Ito + a 7th hue     FAIL -- violet vs blue dE 14.5 for NORMAL vision
  ColorBrewer Dark2         FAIL -- lightness band
  seaborn "colorblind"      FAIL -- chroma and normal-vision floors
  seaborn "muted"           FAIL -- green vs orange dE 4.0 deuteranopia
  the project's old green/red  FAIL -- 2.05:1 contrast, worst pair for red-green CVD
  those hues darkened       FAIL -- orange and amber converge to dE 2.0
"""

# --------------------------------------------------------------------------------------------
# Categorical: identity. Fixed order -- append to the END, never reorder, never cycle.
# --------------------------------------------------------------------------------------------
THEME = [
    "#055E92",   # 1 blue
    "#D47B65",   # 2 vermillion
    "#4DC7A6",   # 3 bluish green
    "#DFBD74",   # 4 amber
    "#A56B8B",   # 5 reddish purple
    "#6CA3C4",   # 6 sky blue
]

NEUTRAL = "#8a8a85"        # "other" / unknown / not-assessable -- deliberately low-chroma
SURFACE = "#ffffff"        # chart surface. Also the separator colour (bar gaps, marker rings),
                           # so it must match the background. The palette was validated against
                           # the near-white #fcfcfb; pure white moves every contrast ratio by
                           # <1%, so the CVD/contrast results below still hold.
INK = "#0b0b0b"            # primary text
INK_SOFT = "#52514e"       # secondary text, axis labels
GRID = "#e4e3df"           # recessive grid

# Sequential: magnitude. ONE hue, light -> dark. Never a rainbow.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Diverging: polarity (correlation matrices, signed margins). Two poles + a NEUTRAL GRAY midpoint,
# so "no relationship" reads as nothing. blue <-> red; never a hue at the midpoint.
DIVERGING = ["#00436b", "#0072B2", "#7fc4e8", "#f0efec", "#ed7a80", "#CE515B", "#9c2026"]
DIVERGING_MID = "#f0efec"


def _assign(keys) -> dict:
    """Map keys onto THEME slots in order. Beyond 8 keys the rest go to NEUTRAL, which is the
    signal to fold them into 'other', facet, or encode the difference some other way."""
    return {k: (THEME[i] if i < len(THEME) else NEUTRAL) for i, k in enumerate(keys)}


# --------------------------------------------------------------------------------------------
# Entity -> colour. One mapping per kind of thing, each in the fixed slot order.
# Colour follows the ENTITY, so a figure that drops a series must not repaint the survivors.
# --------------------------------------------------------------------------------------------
BINDER_CLASS = _assign(["Binder", "Non-Binder", "Mutant+", "Mutant-", "Design", "Target Shuffle"])
BINDER_CLASS["Unknown"] = NEUTRAL

# Six slots, six datasets in the 'rf' set. esm0_top10 takes NEUTRAL: it only ever appears beside
# germinal0 (the 'feasibility' set), so two categories, and grey is unambiguous there.
DATASET = _assign(["peptide", "alphaseq", "benoit", "snir", "germinal0", "esm0"])
DATASET["esm0_top10"] = NEUTRAL
# The DROPPED evaluation sets (see the 'old'/'all' analysis sets). Six slots is the validated
# ceiling and the current benchmark already uses all six, so these take NEUTRAL rather than an
# invented 7th/8th hue. They appear only in the figure that argues for dropping them, where the
# contrast that matters is old-vs-snir-vs-new, not mcmahon-vs-harvey -- colour by that grouping,
# or facet, if the individual sets must be told apart.
DATASET["mcmahon"] = NEUTRAL
DATASET["harvey"] = NEUTRAL
DATASET["germinal"] = NEUTRAL

CATEGORY = _assign(["confidence", "interface", "sequence", "developability", "energy", "geometry"])
CATEGORY["ranking"] = NEUTRAL      # 'ranking' columns are dropped in prep; kept for completeness

# Model FAMILY, not model: 9 models exceed the 8 slots, and the pairs differ only by a setting
# (chai constrained/unconstrained, boltz free/template, esmfold v1/v2). Hue carries the family;
# use a marker or linestyle for the variant.
MODEL_FAMILY = _assign(["af3", "cf", "esmfold2", "chai", "boltz", "pred"])

# DISPLAY labels. The key stays 'pred' -- it is the column prefix (pred_lddt / pred_ilddt) and the
# colour follows that entity -- but 'pred' names no structure predictor, so legends read 'other'.
MODEL_FAMILY_LABEL = {"af3": "AF3", "cf": "CF", "esmfold2": "EF2", "chai": "Chai-1", "boltz": "Boltz-2", "pred": "Other"}

MOL_TYPE = _assign(["nanobody", "antibody"])

# The 'source' column labels the same entities as 'dataset' but spells three of them differently
# (esm/esm0, germinal/germinal0, snir_ab/snir). Colour follows the ENTITY, so they share a colour --
# otherwise the same experiment changes colour between a source-coloured and a dataset-coloured
# figure.
SOURCE = {
    "peptide":  DATASET["peptide"],
    "alphaseq": DATASET["alphaseq"],
    "benoit":   DATASET["benoit"],
    "snir_ab":  DATASET["snir"],
    "esm":      DATASET["esm0"],
    # the three DROPPED evaluation sets, grey as a group (see DATASET above). 'germinal' is grey
    # here too, so a source-coloured figure of a CURRENT set draws its germinal0 design rows grey
    # rather than in germinal0's hue -- they are still separable by the 'dataset' column.
    "germinal": NEUTRAL,
    "mcmahon":  NEUTRAL,
    "harvey":   NEUTRAL,
}

# Feasibility verdict is STATE, not identity: a reserved triplet, never reused as a series.
#
# pass is BLUE, not green. A green/red pair is the intuitive choice and the wrong one: green vs red
# scores dE 5.6 for deuteranopia -- indistinguishable -- against dE 21.9 for this blue/vermillion
# pair (all-pairs PASS). It matters because the filter heatmap has hundreds of cells and cannot
# label each one, so colour IS the only carrier there and must survive colour blindness.
STATUS = {"pass": THEME[0], "fail": THEME[1], "not_assessable": NEUTRAL}


def model_family(model: str) -> str:
    """Collapse a model prefix onto its family, for MODEL_FAMILY lookups."""
    if model.startswith("chai"):
        return "chai"
    if model.startswith("boltz"):
        return "boltz"
    # esmfold2 FIRST: the 'esmfold' prefix also matches it, so the general rule must come second
    # or esmfold2 collapses to a family that MODEL_FAMILY has no key for (-> NEUTRAL grey).
    if model.startswith("esmfold2"):
        return "esmfold2"
    if model.startswith("esmfold"):
        return "esmfold"
    return model


# DISPLAY names for datasets / sources. The data keeps its own spelling (joins, filters and the
# DATASET palette all key on it); only the figure label changes. 'snir_ab' and 'snir' are the same
# experiment spelled differently by the source and dataset columns, and 'esm'/'esm0' likewise.
DATASET_LABEL = {"snir_ab": "Snir", "snir": "Snir", "esm": "ESM0", "esm0": "ESM0",
                 "esm0_top10": "ESM0 top-10"}


def dataset_label(name) -> str:
    """Figure label for a dataset / source: DATASET_LABEL if it has one, else capitalised."""
    return DATASET_LABEL.get(str(name), capitalize_first(name))


def capitalize_first(text) -> str:
    """Capitalise the FIRST character only -- 'category' -> 'Category', while 'PR-AUC', 'af3_iptm'
    and 'snir' keep the rest of their spelling. str.capitalize() would lower-case the remainder."""
    text = str(text)
    return text[:1].upper() + text[1:] if text else text


# Display names for one MODEL, not its family: boltz_free and boltz_template share a colour but
# are different runs. MODEL_FAMILY_LABEL stays the per-family name used by colour legends.
MODEL_LABEL = {"af3": "AF3", "cf": "CF", "esmfold": "EF1", "esmfold2": "EF2",
               "chai_constrained": "Chai-1 (constrained)", "chai_unconstrained": "Chai-1 (unconstrained)",
               "boltz_free": "Boltz-2", "boltz_template": "Boltz-2 (template)", "pred": "Other"}


def model_name(model) -> str:
    """Figure label for one model prefix; falls back to its family's name, then to capitalised."""
    text = str(model)
    if text in MODEL_LABEL:
        return MODEL_LABEL[text]
    return MODEL_FAMILY_LABEL.get(model_family(text), capitalize_first(text))


def model_label(family: str) -> str:
    """Legend text for a model family: the key itself unless MODEL_FAMILY_LABEL renames it."""
    return MODEL_FAMILY_LABEL.get(family, family)


def colors_for(keys, mapping) -> list:
    """Colours for `keys`, unknown -> NEUTRAL. Never zip a colormap over whatever is present:
    that makes colour depend on the filter."""
    return [mapping.get(k, NEUTRAL) for k in keys]


def apply_plot_style() -> None:
    """Uniform figure style. Call once per notebook/script before plotting."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": INK_SOFT,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.labelsize": 15,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,          # grid behind the data, never over it
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "text.color": INK,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.frameon": False,
        "legend.fontsize": 11,
        "legend.title_fontsize": 11,
        "lines.linewidth": 2.0,          # 2px lines
        "lines.markersize": 5,           # >= 8px diameter
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.prop_cycle": mpl.cycler(color=THEME),
    })


def sequential_cmap(name: str = "project_seq"):
    """Single-hue light->dark colormap for magnitude."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, SEQUENTIAL)


def diverging_cmap(name: str = "project_div"):
    """Two-pole colormap with a grey midpoint. Use center=0 / vmin=-vmax so it sits at zero."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, DIVERGING)


# Union for callers that take one `colors_map`; prefer the specific mapping when you know what you
# are colouring. RULE: a figure colours by exactly ONE entity kind -- hues are reused across kinds,
# so encode a second kind with facets, markers or linestyle, never colour.
COLORS_MAP = {**BINDER_CLASS, **DATASET, **SOURCE, **MOL_TYPE}
