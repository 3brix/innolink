"""Design ranking: combine the two rankers (Random Forest + product-composite)
into a consensus shortlist and a disagreement report.

RF is the PRIMARY ranker; the product-composite is complementary. No combined
score is produced -- the shortlist is ordered by the primary method, and the
composite is used only to mark agreement/disagreement.
"""

from .consensus import (
    add_rank,
    build_consensus,
    shortlist,
    disagreements,
    rank_agreement,
)

__all__ = [
    "add_rank",
    "build_consensus",
    "shortlist",
    "disagreements",
    "rank_agreement",
]
