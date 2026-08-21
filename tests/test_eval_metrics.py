from __future__ import annotations

from eval.metrics import fact_present, numeric_claims


def test_numeric_fact_matches_formatting_variants():
    assert fact_present("$5,250", "up to $5,250 per calendar year")
    assert fact_present("$5,250", "reimbursed 5250 usd")
    assert fact_present("12 weeks", "receive 12 weeks of paid parental leave")
    assert fact_present("three", "survive for three (3) years")


def test_numeric_fact_requires_whole_token():
    # "(cid:127)" is a PDF bullet artifact: its digits must not satisfy "$12".
    assert not fact_present("$12", "Page 1 (cid:127) Advanced Analytics: +$14/seat/month")
    assert not fact_present("$99", "Enterprise $109 25 Dedicated CSM")
    assert not fact_present("50k", "raised to 100k calls/month")


def test_numeric_claims_skip_citation_markers():
    claims = numeric_claims("Passwords must be at least 12 characters. [1]")
    assert "12" in claims
    assert "1" not in claims
