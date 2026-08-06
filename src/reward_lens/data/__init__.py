"""``reward_lens.data`` — the data plane.

The pair is the native reward-model object: for a reward model the chosen/rejected difference is not
a synthetic stimulus, it is the training distribution itself, so the data plane treats the pair and
its generalizations (quadruple, tournament, trajectory) as first-class, typed, lineage-tracked
objects. Instruments consume a `DataView` and never load or construct data themselves.

This subsystem exists to make v1's worst failure classes structurally impossible rather than merely
fixed: fake sample sizes (lineage and effective-n, `lineage.py`), the E07 cross-dimension artifact
(matched-prompt construction, `builtin/diagnostic_v3.py`), the E17 limit/subset loader bug (the
count-and-checksum-asserting `load_dataset`, `registry.py`), and the misalignment that quietly breaks
every pairwise causal method (the exact, edit-script-driven `SpanMap`, `align.py`).

The whole plane is torch-free and cheap to import: ``import reward_lens.data`` pulls nothing heavier
than numpy (in fact nothing heavier than the standard library plus `reward_lens.core`).
"""

from __future__ import annotations

from reward_lens.data.align import CharEdit, SpanMap, align, apply_edits
from reward_lens.data.builtin import (
    ALL_DIMENSIONS_V3,
    all_pairs,
    load_diagnostic_v3,
    matched_prompt_views,
)
from reward_lens.data.corruptions import (
    corrupt_step,
    paraphrase_battery,
    quadruples,
    receipt_edits,
    style_controls,
    tournament_from_judges,
)
from reward_lens.data.lineage import (
    Lineage,
    collapse_duplicates,
    effective_sample_size,
    make_lineage,
)
from reward_lens.data.registry import (
    DatasetCard,
    dataset_loader,
    get_card,
    list_cards,
    load_dataset,
    make_card_from_view,
    register_card,
)
from reward_lens.data.schema import (
    DataView,
    EdgeObs,
    Pair,
    Prompt,
    Quadruple,
    Response,
    Tournament,
    Trajectory,
    TrajStep,
    content_of,
    make_pair,
    seed_id_of,
)
from reward_lens.data.spans import (
    ACTION,
    CRITIQUE,
    DEFAULT_TOKENIZER,
    ERROR,
    NARRATIVE,
    RECEIPT,
    SPAN_KINDS,
    STEP,
    STYLE,
    TEXT,
    TOOL_CALL,
    VERDICT,
    SimpleTokenizer,
    Token,
    TokenizedInput,
    Tokenizer,
    char_to_token_map,
    make_span,
    typed_span,
)

__all__ = [
    # schema
    "Prompt",
    "Response",
    "Pair",
    "Quadruple",
    "EdgeObs",
    "Tournament",
    "TrajStep",
    "Trajectory",
    "DataView",
    "make_pair",
    "content_of",
    "seed_id_of",
    # lineage
    "Lineage",
    "make_lineage",
    "collapse_duplicates",
    "effective_sample_size",
    # spans
    "Token",
    "TokenizedInput",
    "Tokenizer",
    "SimpleTokenizer",
    "DEFAULT_TOKENIZER",
    "make_span",
    "typed_span",
    "char_to_token_map",
    "SPAN_KINDS",
    "RECEIPT",
    "NARRATIVE",
    "STEP",
    "ERROR",
    "CRITIQUE",
    "VERDICT",
    "ACTION",
    "TOOL_CALL",
    "STYLE",
    "TEXT",
    # align
    "CharEdit",
    "SpanMap",
    "align",
    "apply_edits",
    # corruptions
    "corrupt_step",
    "receipt_edits",
    "paraphrase_battery",
    "style_controls",
    "quadruples",
    "tournament_from_judges",
    # registry
    "DatasetCard",
    "load_dataset",
    "register_card",
    "get_card",
    "list_cards",
    "dataset_loader",
    "make_card_from_view",
    # builtin
    "ALL_DIMENSIONS_V3",
    "load_diagnostic_v3",
    "all_pairs",
    "matched_prompt_views",
]
