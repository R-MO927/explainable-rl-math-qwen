"""Behavioral tests of record parsing and the leakage-safe splitter on synthetic data.

The real GSM8K files contain no duplicates, so these synthetic cases are the
only way to exercise the exclusion paths.
"""
from src.data.gsm8k import (
    build_splits,
    loose_key,
    near_duplicates,
    parse_raw_record,
    seeded_sample,
)


def rec(split, i, q, ans="5"):
    r = parse_raw_record({"question": q, "answer": f"work\n#### {ans}"}, split, i)
    assert r.error is None, r.error
    return r.record


def test_parse_raw_record_ok_and_verbatim():
    obj = {"question": "  How many?  ", "answer": "2+3=<<2+3=5>>5\n#### 1,005"}
    r = parse_raw_record(obj, "train", 7).record
    assert r["id"] == "gsm8k-train-00007" and r["source_line"] == 8
    assert r["question"] == obj["question"] and r["reference_solution"] == obj["answer"]
    assert r["reference_final_answer"] == "1,005" and r["normalized_final_answer"] == "1005"


def test_parse_raw_record_reports_errors_instead_of_fixing():
    cases = {
        "empty_question": {"question": "  ", "answer": "x #### 1"},
        "empty_answer": {"question": "q", "answer": ""},
        "missing_or_non_string_field": {"question": "q"},
        "bad_final_answer_marker": {"question": "q", "answer": "no marker"},
        "unparseable_final_answer": {"question": "q", "answer": "x #### seven"},
    }
    for reason, obj in cases.items():
        res = parse_raw_record(obj, "train", 0)
        assert res.record is None and res.error.startswith(reason), (reason, res.error)


def test_loose_key_ignores_case_space_punct_but_not_numbers():
    assert loose_key("Tom has 3 apples.") == loose_key("tom  has 3 apples")
    assert loose_key("Tom has 3 apples.") != loose_key("Tom has 4 apples.")


def make_data():
    test = [rec("test", 0, "Ann has 3 cats. How many legs?"), rec("test", 1, "Bob runs 5 km daily.")]
    train = [rec("train", i, f"Unique question number {i} about trains.") for i in range(40)]
    train.append(rec("train", 40, "ann has 3 cats how many legs"))           # leak of test-00000
    train.append(rec("train", 41, "Unique question number 1 about trains!"))  # dup of train-00001
    train.append(rec("train", 42, "Unique question number 2 about trains", ans="9"))  # dup, diff answer
    return train, test


def test_build_splits_excludes_test_leaks_and_train_duplicates():
    train, test = make_data()
    sr = build_splits(train, test, seed=1, val_fraction=0.25)
    reasons = {e["id"]: e for e in sr.excluded}
    assert reasons["gsm8k-train-00040"]["reason"] == "duplicate_of_test"
    assert reasons["gsm8k-train-00040"]["detail"]["test_id"] == "gsm8k-test-00000"
    assert reasons["gsm8k-train-00041"]["reason"] == "duplicate_within_train"
    assert reasons["gsm8k-train-00041"]["detail"] == {"kept_id": "gsm8k-train-00001", "same_final_answer": True}
    assert reasons["gsm8k-train-00042"]["detail"]["same_final_answer"] is False
    # test unchanged; pool partitioned with no overlap and nothing lost
    assert [r["id"] for r in sr.test] == ["gsm8k-test-00000", "gsm8k-test-00001"]
    tr, va = {r["id"] for r in sr.train}, {r["id"] for r in sr.validation}
    assert not tr & va
    assert len(tr) + len(va) + len(sr.excluded) == len(train)
    assert len(va) == 10  # round(40 * 0.25)
    keys = lambda rs: {loose_key(r["question"]) for r in rs}  # noqa: E731
    assert not keys(sr.train) & keys(sr.test) and not keys(sr.validation) & keys(sr.test)
    assert not keys(sr.train) & keys(sr.validation)


def test_build_splits_is_deterministic_and_seed_sensitive():
    train, test = make_data()
    a = build_splits(train, test, seed=1, val_fraction=0.25)
    b = build_splits(list(reversed(train)), test, seed=1, val_fraction=0.25)  # input order must not matter
    c = build_splits(train, test, seed=2, val_fraction=0.25)
    ids = lambda s: [r["id"] for r in s.validation]  # noqa: E731
    assert ids(a) == ids(b)
    assert ids(a) != ids(c)


def test_seeded_sample_stable_subset():
    pool = [f"id{i:03d}" for i in range(100)]
    s1 = seeded_sample(pool, 10, 42, "x")
    assert s1 == seeded_sample(list(reversed(pool)), 10, 42, "x")
    assert set(s1) <= set(pool) and len(set(s1)) == 10
    assert s1 != seeded_sample(pool, 10, 42, "y")


def test_near_duplicates_and_template_masking():
    a = rec("test", 0, "A train travels 270 miles in 3 hours. How long for 180 more miles at that rate?")
    b = rec("train", 0, "A train travels 360 miles in 3 hours. How long for 240 more miles at that rate?")
    c = rec("train", 1, "Completely different problem about baking 12 cookies for a party.")
    plain = near_duplicates([a], [b, c], threshold=0.8)
    masked = near_duplicates([a], [b, c], threshold=0.8, mask_numbers=True)
    assert plain == []                                  # numbers differ -> not a text duplicate
    assert [h["corpus_id"] for h in masked] == ["gsm8k-train-00000"]  # same template detected
