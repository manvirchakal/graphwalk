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


def test_html_to_text_keeps_tables_and_stops_at_references() -> None:
    html = """
    <p>The <b>draft</b> was held.<sup class="reference">[1]</sup></p>
    <h2><span>First round</span><span class="mw-editsection">edit</span></h2>
    <table><tr><th>Pick</th><th>Player</th></tr>
    <tr><td>1</td><td><a>Pat Burrell</a></td></tr></table>
    <h2>References</h2><p>Should not appear.</p>
    """
    assert f.html_to_text(html).splitlines() == [
        "The draft was held.",
        "## First round",
        "Pick | Player",
        "1 | Pat Burrell",
    ]


def test_records_truncate_pages() -> None:
    page = {"evidence": {"pageid": 7, "revid": 1, "title": "A"}}
    questions = [{"id": "q1", "question": "Q?", "answer": {"A": "1"}, "decomposition": [page]}]
    corpus = {"7": {"title": "A", "text": "abcdef"}}
    (record,) = f.records(questions, corpus, max_chars=3)
    assert record["context"] == [["A", ["abc"]]]
    assert record["answer"] == {"A": "1"}
    assert record["type"] == "fanout"
