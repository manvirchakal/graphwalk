from graphwalk.eval.datasets import fanoutqa as f


def test_answer_in_text_matches_the_official_metric() -> None:
    reference = {"Pat Burrell": "Right", "Mark Mulder": "Left"}
    loose, strict = f.answer_in_text(reference, "Pat Burrell: right\nMark Mulder: right")
    assert (loose, strict) == (0.75, False)  # 3 of 4 strings: "Left" is missing
    assert f.answer_in_text(reference, "Pat Burrell: Right | Mark Mulder: Left") == (1.0, True)
    assert f.answer_in_text(["a", "b"], "A only") == (0.5, False)
    assert f.answer_in_text(True, "Yes, it is.") == (1.0, True)
    assert f.answer_in_text(5637000, "about 5,637,000 people") == (1.0, True)
    assert f.answer_in_text("Rose", "Roseville") == (0.0, False)  # word boundaries
    assert f.answer_in_text([], "x") == (0.0, False)


def test_evidence_skips_placeholder_pages() -> None:
    question = {
        "decomposition": [
            {"evidence": {"pageid": 1, "revid": 10, "title": "A"}, "decomposition": []},
            {"evidence": None, "decomposition": [
                {"evidence": {"pageid": "###TBD###", "revid": "###TBD###", "title": "?"}},
                {"evidence": {"pageid": 2, "revid": 20, "title": "B"}},
                {"evidence": {"pageid": 1, "revid": 10, "title": "A"}},
            ]},
        ]
    }  # fmt: skip
    assert [e.title for e in f.evidence(question)] == ["A", "B"]


def test_records_truncate_pages() -> None:
    page = {"evidence": {"pageid": 7, "revid": 1, "title": "A"}}
    questions = [{"id": "q1", "question": "Q?", "answer": {"A": "1"}, "decomposition": [page]}]
    corpus = {"7": {"title": "A", "text": "abcdef"}}
    (record,) = f.records(questions, corpus, max_chars=3)
    assert record["context"] == [["A", ["abc"]]]
    assert record["answer"] == {"A": "1"}
    assert record["type"] == "fanout"
