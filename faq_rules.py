"""FAQ deterministik yang hanya menjawab pertanyaan dengan pola tegas.

Fakta dilengkapi provenance dan field temporal untuk migrasi ke RDF/KG.
Jangan membuat aturan baru tanpa konfirmasi pengelola.
"""
import json
import re
from pathlib import Path

DEFAULT_FAQ_FILE = Path(__file__).resolve().parent / "faq_rules.json"


def _contains(text: str, phrase: str) -> bool:
    return bool(re.search(r"(?<!\w)" + re.escape(phrase.casefold()) + r"(?!\w)", text))


def load_faq(path: Path = DEFAULT_FAQ_FILE) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("rules"), list):
        raise ValueError("Format FAQ tidak dikenal")
    ids = set()
    for rule in data["rules"]:
        if not rule.get("id") or rule["id"] in ids:
            raise ValueError("ID aturan FAQ tidak unik")
        ids.add(rule["id"])
        if rule.get("status") == "active":
            if not rule.get("answer") or not rule.get("match_groups") or not rule.get("scope"):
                raise ValueError(f"Aturan aktif tidak lengkap: {rule['id']}")
    return data


def answer_faq(question: str, ruleset: dict | None = None) -> dict | None:
    """Menghindari matching parsial: seluruh grup intent wajib terpenuhi."""
    data = ruleset or load_faq()
    q = re.sub(r"\s+", " ", question.casefold()).strip()
    matches = []
    for rule in data["rules"]:
        if rule.get("status") != "active":
            continue
        if any(_contains(q, term) for term in rule.get("exclude_terms", [])):
            continue
        if all(any(_contains(q, term) for term in group)
               for group in rule["match_groups"]):
            matches.append(rule)
    if not matches:
        return None
    answers = []
    for rule in matches:
        if rule["answer"] not in answers:
            answers.append(rule["answer"])
    return {
        "answer": "\n\n".join(answers) + (
            f"\n\n*Sumber: {data['source']}, dikonfirmasi {data['confirmed_at']}. "
            "Detail nominal, tanggal, dan syarat khusus tetap harus diverifikasi.*"
        ),
        "rule_ids": [rule["id"] for rule in matches],
        "source": data["source"],
    }
