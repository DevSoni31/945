from __future__ import annotations

import re
import unicodedata

# Common legal suffix expansions — lowercase, no punctuation
_SUFFIXES = {
    "corp": "corporation",
    "inc": "incorporated",
    "ltd": "limited",
    "llc": "limited liability company",
    "pvt": "private",
    "co": "company",
    "intl": "international",
    "assn": "association",
    "dept": "department",
    "govt": "government",
    "natl": "national",
}

# Common address token expansions
_ADDR = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "hwy": "highway",
    "fwy": "freeway",
    "apt": "apartment",
    "ste": "suite",
    "fl": "floor",
    "bldg": "building",
    "dept": "department",
    "opp": "opposite",
    "nr": "near",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
}


def _unicode_normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _lowercase_strip_punct(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)  # punctuation → space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_name(raw: str) -> str:
    if not raw or not raw.strip():
        return ""
    text = _unicode_normalize(raw)
    text = _lowercase_strip_punct(text)
    tokens = text.split()
    tokens = [_SUFFIXES.get(t, t) for t in tokens]
    return " ".join(tokens)


def normalize_address(raw: str) -> str:
    if not raw or not raw.strip():
        return ""
    text = _unicode_normalize(raw)
    text = _lowercase_strip_punct(text)
    tokens = text.split()
    tokens = [_ADDR.get(t, t) for t in tokens]
    return " ".join(tokens)


def combined_text(name: str, address: str) -> str:
    """
    Labelled concatenation for embedding.
    Gracefully handles missing address — does not inject empty tokens.
    """
    n = normalize_name(name)
    a = normalize_address(address)
    if n and a:
        return f"name: {n} address: {a}"
    if n:
        return f"name: {n}"
    if a:
        return f"address: {a}"
    return ""
