"""Tests for finding a company's job board without guessing its slug."""

from __future__ import annotations

import httpx2

from job_hunters.discover import probe, slug_variants


def test_slug_variants_cover_the_spellings_the_watchlist_needed() -> None:
    """These six non-obvious slugs must all be reachable."""
    assert "scaleai" in slug_variants("Scale")
    assert "togetherai" in slug_variants("Together")
    assert "liquid-ai" in slug_variants("Liquid")
    assert "lumaai" in slug_variants("Luma")
    assert "perplexity" in slug_variants("Perplexity AI")
    assert "arizeai" in slug_variants("Arize")


def test_slug_variants_are_unique_and_start_with_the_obvious_one() -> None:
    """The most likely slug comes first and no variant is listed twice."""
    variants = slug_variants("Scale AI")
    assert variants[0] == "scaleai"
    assert len(variants) == len(set(variants))


def test_probe_reports_only_boards_that_answer_with_the_right_shape() -> None:
    """Only boards answering with the shape their ATS actually returns count as hits."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        """Answers each ATS's probe URL with a plausible payload, else 404."""
        url = str(request.url)
        if "greenhouse" in url and "/boards/acme/" in url:
            return httpx2.Response(200, json={"jobs": [{"id": 1}, {"id": 2}]})
        if "lever" in url and "/postings/acme?" in url:
            return httpx2.Response(200, json=[{"id": "x"}])
        if "ashby" in url:
            return httpx2.Response(200, json={"jobs": []})  # ashby's "unknown board"
        return httpx2.Response(404, json={"error": "not found"})

    hits = probe("Acme", client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    by_ats = {h.ats: h for h in hits if h.token == "acme"}
    assert by_ats["greenhouse"].job_count == 2
    assert by_ats["lever"].job_count == 1
    assert by_ats["ashby"].job_count == 0
    assert "tier: discovered" in by_ats["greenhouse"].watchlist_line("acme", "Acme")


def test_probe_with_nothing_found_returns_empty() -> None:
    """When every board answers 404, probing returns no hits at all."""
    client = httpx2.Client(transport=httpx2.MockTransport(lambda r: httpx2.Response(404)))
    assert probe("Nobody", client=client) == []
