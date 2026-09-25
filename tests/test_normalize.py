from er945.normalize import normalize_name, normalize_address, combined_text


def test_suffix_expansion():
    assert "corporation" in normalize_name("Acme Corp")
    assert "limited" in normalize_name("Tata Ltd")
    assert "private" in normalize_name("XYZ Pvt Ltd")


def test_address_abbreviation():
    assert "road" in normalize_address("MG Rd")
    assert "street" in normalize_address("Baker St")
    assert "avenue" in normalize_address("5th Ave")


def test_punctuation_stripped():
    assert "," not in normalize_name("Smith, Jones & Co.")
    assert "&" not in normalize_name("Smith & Jones")


def test_empty_name_returns_empty():
    assert normalize_name("") == ""
    assert normalize_name("   ") == ""


def test_empty_address_returns_empty():
    assert normalize_address("") == ""


def test_combined_both_present():
    out = combined_text("Acme Corp", "MG Rd")
    assert out.startswith("name:")
    assert "address:" in out


def test_combined_missing_address():
    out = combined_text("Acme Corp", "")
    assert "name:" in out
    assert "address:" not in out


def test_combined_both_missing():
    assert combined_text("", "") == ""


def test_unicode_normalized():
    # Full-width characters should collapse
    out = normalize_name("ＡＣＭＥ")
    assert out == "acme"
