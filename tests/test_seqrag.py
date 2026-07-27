import importlib
import sys
import types
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class _DummyNLP:
    @property
    def sents(self):
        return []

    def __call__(self, text):
        return self


class _DummyCounter:
    def __init__(self):
        self.reflect = 0

    def add_generate(self, *_args, **_kwargs):
        return None


class _DummyBasicRAG:
    pass


def _load_seqrag_with_stubs():
    sys.modules.pop("SeqRAG", None)
    sys.modules["spacy"] = types.SimpleNamespace(load=lambda _: _DummyNLP())
    sys.modules["examples"] = types.SimpleNamespace(
        TUTOR_ADVICE_EXAMPLES=[],
        REFLECT_EXAMPLES=[],
    )
    sys.modules["prompts"] = types.SimpleNamespace(
        ENTITY_REPLEACE_TEMPLATE="{question} {sentence}",
        KEYWORDS_TEMPLATE="{sentence}",
        MISSING_INFO_TEMPLATE="{question} {response}",
        TUTOR_ADVICE_HEADER="",
        TUTOR_USE_DOCS="",
        TUTOR_USE_DOCS_MIDDLE="",
        TUTOR_NOT_USE_DOCS_MIDDLE="",
        TUTOR_ADVICE_MIDDLE="",
        ADVICE_TEMPLATE="{question}",
        REFLECTION_HEADER="",
        REFLECT_USE_DOC="",
        REFLECT_USE_DOC_MIDDLE="",
        REFLECT_NOT_USE_DOC_MIDDLE="",
        REFLECTION_MIDDLE="",
        REFLECTION_TEMPLATE="{question}",
    )
    sys.modules["utils"] = types.SimpleNamespace(
        split_sentences=lambda text: [text] if text else [],
        find_element_index=lambda seq, target: next((i for i, v in enumerate(seq) if v == target), -1),
        is_ans_unknown=lambda _answers: (None, False),
        get_reason_prompt=lambda *_args, **_kwargs: "",
        get_answer_prompt=lambda *_args, **_kwargs: "",
        get_docstr=lambda _docs: "",
        _coerce_text_value=lambda value: "" if value is None else str(value),
        List=list,
    )
    sys.modules["rag"] = types.SimpleNamespace(BasicRAG=_DummyBasicRAG)
    return importlib.import_module("SeqRAG")


def _build_rag(**attrs):
    seqrag = _load_seqrag_with_stubs()
    rag = seqrag.SeqConfidenceRAG.__new__(seqrag.SeqConfidenceRAG)
    rag.use_reflect = attrs.pop("use_reflect", True)
    rag.disable_retrieval = attrs.pop("disable_retrieval", False)
    rag.experiment_condition = attrs.pop("experiment_condition", None)
    for key, value in attrs.items():
        setattr(rag, key, value)
    return rag


def test_default_strategy_keeps_legacy_behavior():
    rag = _build_rag()

    assert rag._should_retrieve_for_level(-1) is True
    assert rag._should_retrieve_for_level(0) is False
    assert rag._should_retrieve_for_level(1) is False
    assert rag._should_reflect_for_level(0) is True
    assert rag._should_reflect_for_level(-1) is False


def test_baseline_condition_disables_retrieval_and_reflection():
    rag = _build_rag(experiment_condition="baseline")

    for level in (-1, 0, 1):
        assert rag._should_retrieve_for_level(level) is False
        assert rag._should_reflect_for_level(level) is False


def test_low_only_condition_retrieves_only_low_confidence():
    rag = _build_rag(experiment_condition="low_only")

    assert rag._should_retrieve_for_level(-1) is True
    assert rag._should_retrieve_for_level(0) is False
    assert rag._should_retrieve_for_level(1) is False
    assert rag._should_reflect_for_level(0) is False


def test_mid_only_condition_retrieves_only_mid_confidence_without_reflection():
    rag = _build_rag(experiment_condition="mid_only")

    assert rag._should_retrieve_for_level(-1) is False
    assert rag._should_retrieve_for_level(0) is True
    assert rag._should_retrieve_for_level(1) is False
    assert rag._should_reflect_for_level(0) is False


def test_high_only_condition_retrieves_only_high_confidence():
    rag = _build_rag(experiment_condition="high_only")

    assert rag._should_retrieve_for_level(-1) is False
    assert rag._should_retrieve_for_level(0) is False
    assert rag._should_retrieve_for_level(1) is True
    assert rag._should_reflect_for_level(1) is False
