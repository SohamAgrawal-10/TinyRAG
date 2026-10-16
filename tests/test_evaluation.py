
import pytest
from tinyrag.evaluation import precision_at_k, recall_at_k, reciprocal_rank
from tinyrag.eval_runner import is_out_of_scope


def test_precision_at_k():
    retrieved = ["a", "b", "c"]
    relevant = ["a", "c"]

    assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)


def test_recall_at_k():
    retrieved = ["a", "b", "c"]
    relevant = ["a", "c", "d"]

    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)


def test_reciprocal_rank():
    retrieved = ["a", "b", "c"]
    relevant = ["c"]

    assert reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 3)


def test_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k(["a"], ["a"], k=0)


def test_recall_at_k_with_no_relevant_chunks():
    assert recall_at_k(["a", "b"], [], k=2) == 0.0


def test_reciprocal_rank_when_no_relevant_chunk_is_retrieved():
    assert reciprocal_rank(["a", "b", "c"], ["z"]) == 0.0


def test_rejects_score_below_threshold():
    assert is_out_of_scope(0.1384, 0.30)


def test_keeps_score_above_threshold():
    assert not is_out_of_scope(0.5959, 0.30)


def test_rejects_when_no_results():
    assert is_out_of_scope(None, 0.30)


def test_keeps_score_equal_to_threshold():
    assert not is_out_of_scope(0.30, 0.30)
