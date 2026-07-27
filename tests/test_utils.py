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


def _load_utils_with_stubbed_spacy():
    sys.modules.pop("utils", None)
    sys.modules["spacy"] = types.SimpleNamespace(load=lambda _: _DummyNLP())
    return importlib.import_module("utils")


def test_is_ans_unknown_returns_false_for_empty_answers():
    utils = _load_utils_with_stubbed_spacy()

    assert utils.is_ans_unknown([]) == (None, False)


def test_process_reflect_text_extracts_string_from_nested_modified_object():
    utils = _load_utils_with_stubbed_spacy()

    raw_text = '{"Modified": {"text": "Shirley Temple later served as U.S. Ambassador to Ghana."}}'

    assert utils.process_reflect_text(raw_text) == "Shirley Temple later served as U.S. Ambassador to Ghana."


def test_process_answer_text_extracts_string_from_nested_continue_object():
    utils = _load_utils_with_stubbed_spacy()

    raw_text = '{"continue": {"text": "So, the answer is Shirley Temple."}}'

    assert utils.process_answer_text(raw_text, pre_answer="") == "So, the answer is Shirley Temple."
