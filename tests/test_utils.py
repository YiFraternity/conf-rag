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
