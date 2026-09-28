from graphwalk.core.hashing import content_hash

EMPTY_SHA256 = "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_known_value() -> None:
    assert content_hash("") == EMPTY_SHA256
    assert content_hash(b"") == EMPTY_SHA256


def test_str_and_utf8_bytes_hash_equal() -> None:
    assert content_hash("héllo") == content_hash("héllo".encode())


def test_json_is_canonical_over_key_order() -> None:
    assert content_hash({"a": 1, "b": [1, 2]}) == content_hash({"b": [1, 2], "a": 1})


def test_json_distinguishes_int_float_bool() -> None:
    assert len({content_hash(1), content_hash(1.0), content_hash(True)}) == 3


def test_json_string_differs_from_raw_text() -> None:
    # A JSON value is hashed as JSON; a bare str is hashed as text.
    assert content_hash(["x"]) != content_hash("x")
