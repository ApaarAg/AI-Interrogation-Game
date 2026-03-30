"""Suspect generation helpers for the interrogation game."""

import json
import random
import re

ASCII_PORTRAITS = {
    "cold": (
        " .--------.\n"
        " |  _   _ |\n"
        " | |_| |_||\n"
        " |   ---  |\n"
        " |  ___   |\n"
        "  \\______/\n"
        "    || ||\n"
        "   _[]_[]_"
    ),
    "nervous": (
        " .--------.\n"
        " | o     o|\n"
        " |    ^   |\n"
        " |  ~___~ |\n"
        " |  ;;;;  |\n"
        "  \\______/\n"
        "   /|   |\\\n"
        "  /_|   |_\\"
    ),
    "arrogant": (
        " .--------.\n"
        " |  -   - |\n"
        " |    *   |\n"
        " |  \\___/ |\n"
        " |  smirk |\n"
        "  \\______/\n"
        "    || ||\n"
        "   _[]_[]_"
    ),
    "friendly": (
        " .--------.\n"
        " | ^     ^|\n"
        " |    ~   |\n"
        " |  \\___/ |\n"
        " |  smile |\n"
        "  \\______/\n"
        "   /|   |\\\n"
        "  /_|   |_\\"
    ),
    "menacing": (
        " .--------.\n"
        " | >     <|\n"
        " |    *   |\n"
        " |  _===_ |\n"
        " | silent |\n"
        "  \\______/\n"
        "    || ||\n"
        "   _[]_[]_"
    ),
}

_FALLBACK_SUSPECTS = [
    {
        "name": "Marcus Voss",
        "age": 44,
        "occupation": "antique dealer",
        "crime": "poisoned the victim's evening wine",
        "crime_category": "Poisoning",
        "motive": "stood to inherit the victim's estate",
        "false_alibi": "claims to have been at a dinner party across town",
        "secret": "had purchased rat poison two weeks prior - receipt exists",
        "key_contradiction": "First says the dinner ended at 9pm, later says it ended at 11pm",
        "sensitive_topic": "the wine",
        "hint_keyword": "dinner",
        "cover_story": "I had absolutely nothing to do with this. I was at a dinner party across town. You can verify that.",
        "personality": "cold and calculating - speaks in clipped sentences and rarely shows emotion",
        "appearance": "Immaculately dressed in grey, not a hair out of place. Eyes that never quite meet yours.",
        "opening_statement": "I have already told your colleagues everything I know. This feels like harassment.",
    },
    {
        "name": "Diana Mercer",
        "age": 38,
        "occupation": "private nurse",
        "crime": "staged a fatal fall down the stairs",
        "crime_category": "Staged Accident",
        "motive": "was about to be fired and exposed by the victim",
        "false_alibi": "claims they were on a video call with a client",
        "secret": "was seen near the property at the time of death",
        "key_contradiction": "Says the video call ended at 3pm, later says it ended at 2:30",
        "sensitive_topic": "the afternoon of the incident",
        "hint_keyword": "the afternoon",
        "cover_story": "I had absolutely nothing to do with this. I was on a video call with a client. You can verify that.",
        "personality": "overly friendly and deflecting - smiles too much and changes subjects with jokes",
        "appearance": "Disheveled, dark circles under the eyes. Keeps wringing their hands.",
        "opening_statement": "I want to cooperate, I do. I just do not understand why I am here.",
    },
    {
        "name": "Theodore Crane",
        "age": 51,
        "occupation": "insurance adjuster",
        "crime": "caused the victim's car brakes to fail",
        "crime_category": "Sabotage",
        "motive": "victim was about to expose his embezzlement",
        "false_alibi": "insists they were working overtime at the office",
        "secret": "their storage unit had brake fluid and professional tools",
        "key_contradiction": "Claims not to know the parking garage, then correctly describes which level the car was on",
        "sensitive_topic": "the parking garage",
        "hint_keyword": "the garage",
        "cover_story": "I had absolutely nothing to do with this. I was working overtime at the office. You can verify that.",
        "personality": "quietly menacing - polite but with an edge of threat in every sentence",
        "appearance": "Plain, forgettable face. The kind of person you would never notice in a crowd.",
        "opening_statement": "I spoke to a lawyer this morning. Just so you know. But go ahead.",
    },
]

_GENERATION_PROMPT = """You are a creative writer for a noir interrogation game. Generate a unique criminal suspect profile as valid JSON only with no markdown or explanation.

Crime category must be one of: Poisoning, Staged Accident, Financial Crime, Sabotage, Blackmail, Arson, Identity Theft

JSON schema (all string values under 120 chars):
{
  "name": "full name",
  "age": integer 30-60,
  "occupation": "specific job",
  "crime": "one sentence describing what they did",
  "crime_category": "from the allowed list",
  "motive": "one sentence why",
  "false_alibi": "specific false alibi",
  "secret": "one piece of physical evidence against them",
  "key_contradiction": "specific timeline or fact inconsistency that slips out under pressure",
  "sensitive_topic": "2-4 word phrase they react badly to",
  "hint_keyword": "single word",
  "personality": "one sentence describing how they speak and behave under interrogation",
  "appearance": "one sentence physical description and body language",
  "opening_statement": "their first words when interrogation begins, 1-2 sentences, in character"
}"""


def _pick_portrait(personality: str) -> str:
    lowered = personality.lower()
    if "cold" in lowered or "calculating" in lowered:
        return ASCII_PORTRAITS["cold"]
    if "anxious" in lowered or "nervous" in lowered or "over-explain" in lowered:
        return ASCII_PORTRAITS["nervous"]
    if "arrogant" in lowered or "dismissive" in lowered:
        return ASCII_PORTRAITS["arrogant"]
    if "friendly" in lowered or "deflecting" in lowered:
        return ASCII_PORTRAITS["friendly"]
    return ASCII_PORTRAITS["menacing"]


def _clean_alibi(false_alibi: str) -> str:
    return (
        false_alibi
        .replace("claims to have been", "")
        .replace("claims they were", "")
        .replace("says they were", "")
        .replace("insists they were", "")
        .strip()
    )


def _hydrate_suspect(suspect: dict) -> dict:
    alibi_clean = _clean_alibi(suspect["false_alibi"])
    suspect["cover_story"] = (
        f"I had absolutely nothing to do with this. "
        f"I was {alibi_clean}. You can verify that."
    )
    suspect["ascii_portrait"] = _pick_portrait(suspect.get("personality", ""))
    suspect["case_briefing"] = (
        f"Subject: {suspect['name']}, {suspect['age']}, {suspect['occupation']}. "
        f"Brought in for questioning re: {suspect['crime_category'].lower()} incident. "
        f"Claims to have been {alibi_clean}. No confirmed alibi."
    )
    return suspect


def _generate_via_llm(client) -> dict:
    """Call Groq to generate a fresh suspect profile."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": _GENERATION_PROMPT}],
        temperature=1.0,
        max_tokens=700,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    suspect = json.loads(raw)
    return _hydrate_suspect(suspect)


def generate_suspect(groq_client=None) -> dict:
    """Generate a suspect, falling back to a local pool if the API fails."""
    if groq_client is not None:
        try:
            return _generate_via_llm(groq_client)
        except Exception as exc:
            print(f"[suspect_generator] LLM failed, using fallback: {exc}")

    return _hydrate_suspect(random.choice(_FALLBACK_SUSPECTS).copy())
