import pytest

from er945.score import entity_f05, macro_f05


def test_correct_singleton():
    assert entity_f05(set(), set()) == 1.0


def test_false_match_on_singleton():
    assert entity_f05(set(), {"S2-1"}) == 0.0


def test_missed_match():
    assert entity_f05({"S2-1"}, set()) == 0.0


def test_exact_match():
    assert entity_f05({"S2-1", "S3-1"}, {"S2-1", "S3-1"}) == 1.0


def test_extra_match():
    score = entity_f05(
        {"S2-1", "S3-1"},
        {"S2-1", "S3-1", "S2-2"},
    )
    assert score == pytest.approx(5 / 7)


def test_macro_includes_singletons():
    truth = {"S1-1": set(), "S1-2": {"S2-1"}}
    predicted = {"S1-1": set(), "S1-2": set()}
    assert macro_f05(truth, predicted) == 0.5


def test_missing_source1_is_an_error():
    with pytest.raises(ValueError, match="missing"):
        macro_f05({"S1-1": set()}, {})
