"""Evidence labels and deterministic guardrails for generative RAG answers.

This checks structural provenance and unsupported numbers/links; it is NOT a
complete natural-language entailment verifier. Curated FAQ is authoritative
for the rules it covers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from institutional_policy import POLICY_TEXT

POLICY_REF = "P1"
REF_RE = re.compile(r"\[(S\d+|P1)\]")
NUMBER_RE = re.compile(r"(?<![A-Za-z])\d[\d.,:/-]*")
URL_RE = re.compile(r"https?://[^\s<>)\]]+")


@dataclass
class Evidence:
    text: str
    sources: dict[str, dict]


def make_evidence(contexts: list[dict], live_context: dict | None = None) -> Evidence:
    sources = {
        POLICY_REF: {
            "title": "Kebijakan operasional terkonfirmasi pengelola",
            "url": None,
            "text": POLICY_TEXT,
            "type": "policy",
        }
    }
    parts = [f"[P1] KEBIJAKAN PUSAT BAHASA (konfirmasi pengelola):\n{POLICY_TEXT}"]
    chosen = list(contexts[:6])
    if live_context is not None:
        chosen.append(live_context)
    for number, context in enumerate(chosen, 1):
        ref = f"S{number}"
        content = context["text"][:2500]
        sources[ref] = {
            "title": context["title"],
            "url": context["url"],
            "text": content,
            "type": "website",
        }
        parts.append(
            f"[{ref}] {context['title']}; URL: {context['url']}; "
            f"terakhir diperiksa: {context.get('checked_at', 'tidak diketahui')}\n{content}"
        )
    return Evidence(text="\n\n".join(parts), sources=sources)


@dataclass
class GuardResult:
    ok: bool
    message: str
    referenced: list[str]


def validate_answer(answer: str, evidence: Evidence) -> GuardResult:
    """Reject unknown citations, unsourced numbers, and fabricated external links."""
    citations = list(dict.fromkeys(REF_RE.findall(answer)))
    if not citations:
        return GuardResult(False, "Jawaban tidak mencantumkan bukti [S1] atau [P1].", [])
    if any(ref not in evidence.sources for ref in citations):
        return GuardResult(False, "Jawaban mencantumkan sumber yang tidak ditemukan.", [])
    cited_text = " ".join(evidence.sources[ref]["text"] for ref in citations)
    # Explicit dates/fees/scores must be present in cited evidence (or policy).
    plain_answer = REF_RE.sub("", answer)
    for numeric in NUMBER_RE.findall(plain_answer):
        numeric = numeric.rstrip(".,:/-")
        if not numeric:
            continue
        if not any(numeric in evidence.sources[ref]["text"] for ref in citations):
            return GuardResult(False, "Angka pada jawaban tidak ditemukan dalam bukti yang dikutip.", citations)
    urls = set(URL_RE.findall(plain_answer))
    known_urls = {src["url"] for src in evidence.sources.values() if src["url"]}
    if urls - known_urls:
        return GuardResult(False, "Tautan yang ditulis tidak ada dalam sumber terverifikasi.", citations)
    return GuardResult(True, "OK", citations)
