"""Helpers for scoring a player's accusation."""

import json
import re

_JUDGE_PROMPT = """You are an impartial judge for a noir detective game. A player has interrogated a suspect and made an accusation. Score their accusation against the hidden truth.

Hidden truth:
- Crime: {crime}
- Motive: {motive}
- Key secret/evidence: {secret}
- Key contradiction: {key_contradiction}

Player's accusation:
- Crime stated: {player_crime}
- Motive stated: {player_motive}
- Evidence/observations cited: {player_evidence}

Score generously for semantic understanding, not exact wording. If the player captures the spirit of the crime or motive correctly, award full points. Partial credit for being close.

Return only valid JSON, no markdown:
{{
  "crime_score": integer 0-35,
  "crime_correct": true or false,
  "crime_reason": "one sentence explanation",
  "motive_score": integer 0-35,
  "motive_correct": true or false,
  "motive_reason": "one sentence explanation",
  "evidence_score": integer 0-15,
  "evidence_reason": "one sentence explanation",
  "secret_cited": true or false,
  "contradiction_spotted": true or false
}}"""


def _llm_evaluate(client, accusation: dict, suspect: dict) -> dict:
    """Ask the model to score the accusation."""
    prompt = _JUDGE_PROMPT.format(
        crime=suspect["crime"],
        motive=suspect["motive"],
        secret=suspect["secret"],
        key_contradiction=suspect["key_contradiction"],
        player_crime=accusation.get("crime", ""),
        player_motive=accusation.get("motive", ""),
        player_evidence=accusation.get("evidence", "") or "None provided",
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=400,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    return json.loads(raw)


def _keyword_match(player_text: str, truth_text: str, keywords: list | None = None) -> bool:
    player_lower = player_text.lower()
    truth_lower = truth_text.lower()
    truth_words = {word for word in re.split(r"\W+", truth_lower) if len(word) > 3}
    player_words = {word for word in re.split(r"\W+", player_lower) if len(word) > 3}

    if keywords:
        for keyword in keywords:
            if keyword.lower() in player_lower:
                return True

    return len(truth_words & player_words) >= 2


def _keyword_evaluate(accusation: dict, suspect: dict) -> dict:
    """Fallback scorer used if the LLM call fails."""
    crime_keywords = {
        "Poisoning": ["poison", "wine", "drink", "toxin"],
        "Staged Accident": ["fall", "staged", "stairs", "pushed", "accident"],
        "Financial Crime": ["forge", "document", "inheritance", "fraud", "will"],
        "Sabotage": ["brake", "car", "sabotage", "tamper"],
        "Blackmail": ["blackmail", "secret", "threaten", "exposure"],
        "Arson": ["fire", "arson", "burn", "accelerant"],
        "Identity Theft": ["identity", "fraud", "impersonate", "document"],
    }

    crime_hit = _keyword_match(
        accusation["crime"],
        suspect["crime"],
        crime_keywords.get(suspect["crime_category"], []),
    )
    motive_hit = _keyword_match(accusation["motive"], suspect["motive"])

    crime_score = 35 if crime_hit else 0
    motive_score = 35 if motive_hit else 0

    evidence_text = accusation.get("evidence", "").lower()
    secret_words = {word for word in re.split(r"\W+", suspect["secret"].lower()) if len(word) > 3}
    contradiction_words = {
        word for word in re.split(r"\W+", suspect["key_contradiction"].lower()) if len(word) > 3
    }
    evidence_words = {word for word in re.split(r"\W+", evidence_text) if len(word) > 3}

    evidence_score = 0
    secret_cited = len(secret_words & evidence_words) >= 2
    contradiction_spotted = len(contradiction_words & evidence_words) >= 2

    if secret_cited:
        evidence_score += 8
    if contradiction_spotted:
        evidence_score += 7
    if evidence_score == 0 and len(evidence_text) > 20:
        evidence_score = 5

    return {
        "crime_score": crime_score,
        "crime_correct": crime_hit,
        "crime_reason": "Keyword match" if crime_hit else "No keyword match found",
        "motive_score": motive_score,
        "motive_correct": motive_hit,
        "motive_reason": "Keyword match" if motive_hit else "No keyword match found",
        "evidence_score": evidence_score,
        "evidence_reason": "Based on keyword overlap with secret or contradiction",
        "secret_cited": secret_cited,
        "contradiction_spotted": contradiction_spotted,
    }


def evaluate_accusation(
    accusation: dict,
    suspect: dict,
    message_count: int,
    elapsed_seconds: float,
    groq_client=None,
    confession_bonus: int = 0,
) -> dict:
    """Score the accusation using the LLM first, then a keyword fallback."""
    if groq_client is not None:
        try:
            score_data = _llm_evaluate(groq_client, accusation, suspect)
        except Exception as exc:
            print(f"[scoring] LLM evaluation failed, using keyword fallback: {exc}")
            score_data = _keyword_evaluate(accusation, suspect)
    else:
        score_data = _keyword_evaluate(accusation, suspect)

    crime_score = max(0, min(35, int(score_data.get("crime_score", 0))))
    motive_score = max(0, min(35, int(score_data.get("motive_score", 0))))
    evidence_score = max(0, min(15, int(score_data.get("evidence_score", 0))))

    crime_correct = bool(score_data.get("crime_correct", False))
    motive_correct = bool(score_data.get("motive_correct", False))
    secret_cited = bool(score_data.get("secret_cited", False))
    contradiction_spotted = bool(score_data.get("contradiction_spotted", False))

    crime_reason = score_data.get("crime_reason", "")
    motive_reason = score_data.get("motive_reason", "")
    evidence_reason = score_data.get("evidence_reason", "")

    if message_count <= 8:
        efficiency_score = 10
        efficiency_label = f"Sharp interrogation ({message_count} questions)"
    elif message_count <= 15:
        efficiency_score = 7
        efficiency_label = f"Solid interrogation ({message_count} questions)"
    elif message_count <= 25:
        efficiency_score = 4
        efficiency_label = f"Lengthy interrogation ({message_count} questions)"
    else:
        efficiency_score = 0
        efficiency_label = f"Very long interrogation ({message_count} questions)"

    breakdown = [
        {
            "label": f"Crime identified - {crime_reason}" if crime_reason else "Crime identified",
            "points": f"+{crime_score}",
            "correct": crime_correct,
        },
        {
            "label": f"Motive identified - {motive_reason}" if motive_reason else "Motive identified",
            "points": f"+{motive_score}",
            "correct": motive_correct,
        },
    ]

    if secret_cited:
        breakdown.append({"label": "Key evidence cited", "points": "+8", "correct": True})
    if contradiction_spotted:
        breakdown.append({"label": "Contradiction spotted", "points": "+7", "correct": True})
    if evidence_score > 0 and not secret_cited and not contradiction_spotted:
        breakdown.append({
            "label": f"Evidence attempted - {evidence_reason}",
            "points": f"+{evidence_score}",
            "correct": True,
        })
    if evidence_score == 0:
        breakdown.append({"label": "No useful evidence cited", "points": "+0", "correct": False})

    breakdown.append({
        "label": efficiency_label,
        "points": f"+{efficiency_score}",
        "correct": efficiency_score > 0,
    })

    total = crime_score + motive_score + evidence_score + efficiency_score
    if confession_bonus > 0:
        breakdown.insert(0, {
            "label": "Broke the suspect - confession extracted",
            "points": f"+{confession_bonus}",
            "correct": True,
        })
        total += confession_bonus
    total = min(100, total)

    if total >= 85:
        verdict = "CASE CLOSED"
        verdict_detail = "Brilliant detective work. Every crack in the story led you straight to the truth."
        verdict_class = "perfect"
    elif total >= 65:
        verdict = "STRONG CASE"
        verdict_detail = "You found the key threads. The prosecution has what it needs."
        verdict_class = "good"
    elif total >= 40:
        verdict = "PARTIAL TRUTH"
        verdict_detail = "You sensed something was wrong, but the full picture eluded you."
        verdict_class = "partial"
    else:
        verdict = "CASE UNSOLVED"
        verdict_detail = "The suspect walks free. They were more careful than you were thorough."
        verdict_class = "fail"

    elapsed_minutes = int(elapsed_seconds // 60)
    elapsed_remainder = int(elapsed_seconds % 60)

    return {
        "score": total,
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "verdict_class": verdict_class,
        "breakdown": breakdown,
        "time_taken": f"{elapsed_minutes}m {elapsed_remainder}s",
        "message_count": message_count,
    }
