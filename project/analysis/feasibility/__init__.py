from analysis.feasibility.gates import DEFAULT_GATES
from analysis.feasibility.filters import apply_gates, feasibility_summary
from analysis.feasibility.responsiveness import responsiveness, consistency

# NOTE: responsiveness()/consistency() compute per-dataset Cliff's-delta effect sizes and
# cross-dataset sign-consistency (a metric-generalisation signal). They are kept as a
# library for analysis/notebooks but are not yet wired into a pipeline stage.

__all__ = ["DEFAULT_GATES", "apply_gates", "feasibility_summary",
           "responsiveness", "consistency"]
