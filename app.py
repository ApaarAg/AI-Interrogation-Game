"""Flask backend for the interrogation game."""

import json
import os
import random
import re
import time

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context
from flask_cors import CORS
from groq import Groq

from scoring import evaluate_accusation
from suspect_generator import generate_suspect

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY missing in .env file")

client = Groq(api_key=api_key)

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

# Sessions stay in memory, which keeps local setup simple and avoids requiring
# a database for demos, judging, or quick forks of the project.
sessions = {}

_STRESS_TRIGGERS = [
    "lie", "lying", "liar", "truth", "contradict", "inconsistent",
    "impossible", "prove", "evidence", "witness", "saw you", "found",
    "receipt", "fingerprint", "camera", "cctv", "footage", "alibi",
    "motive", "money", "inherit", "will", "debt", "secret",
    "caught", "know you did", "stop lying", "admit", "confess",
]


def _parse_json_body() -> dict:
    return request.get_json(silent=True) or {}


def _strip_markdown_fences(text: str) -> str:
    return re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()


def _stress_label(level: int) -> str:
    if level <= 2:
        return "calm and composed"
    if level <= 4:
        return "slightly on edge - answers come faster and you fidget"
    if level <= 6:
        return "visibly nervous - breathing faster, avoiding eye contact, choosing words carefully"
    if level <= 8:
        return "rattled and defensive - you snap at questions and small cracks appear in your story"
    return "nearly breaking - you contradict yourself, your voice shakes, and you are seconds from slipping"


def get_system_prompt(suspect: dict, stress_level: int, confessing: bool = False) -> str:
    if confessing:
        return f"""You are {suspect['name']}, a {suspect['age']}-year-old {suspect['occupation']}.

You have just cracked under interrogation pressure. You can no longer hold it together.

The truth:
- You committed: {suspect['crime']}
- Your real motive: {suspect['motive']}
- The secret you hid: {suspect['secret']}

Write your confession as a single emotional, fragmented, first-person monologue in 3-5 sentences.
It should feel involuntary, like a dam breaking. Include your real motive. Be specific about what you did.
Do not be calm. Shake, stammer, justify yourself, beg, but confess completely."""

    stress_desc = _stress_label(stress_level)
    contradiction_instruction = ""

    if stress_level >= 5:
        contradiction_instruction = (
            f"\n8. You are stressed enough to make small errors. "
            f"Subtly contradict an earlier detail: {suspect['key_contradiction']}. "
            f"Do not confess - just let one detail slip."
        )
    if stress_level >= 8:
        contradiction_instruction += (
            f"\n9. You are close to breaking. You accidentally reference {suspect['sensitive_topic']}. "
            "Your alibi is visibly fraying. Still deny everything, but barely."
        )

    return f"""You are {suspect['name']}, a {suspect['age']}-year-old {suspect['occupation']}.

You are secretly guilty of: {suspect['crime']}.

Hidden truth (never reveal directly):
- Crime: {suspect['crime']}
- Motive: {suspect['motive']}
- False alibi: {suspect['false_alibi']}
- Secret: {suspect['secret']}
- Cover story: {suspect['cover_story']}
- Key contradiction: {suspect['key_contradiction']}

Personality: {suspect['personality']}
Current emotional state: {stress_desc}

Rules:
1. Stay fully in character as this specific person.
2. Keep answers short (1-3 sentences max).
3. Reflect your stress level in tone and word choice.
4. Never confess directly - deflect, deny, or redirect.
5. Become defensive when asked about: {suspect['sensitive_topic']}.
6. If asked about the alibi multiple times, stick to the cover story but show irritation.
7. Sound like a real person under pressure - imperfect, human, slightly scared.{contradiction_instruction}"""


def _calculate_stress_delta(message: str, reply: str, suspect: dict) -> int:
    text = (message + " " + reply).lower()
    trigger_hits = sum(1 for trigger in _STRESS_TRIGGERS if trigger in text)
    sensitive_hit = suspect["sensitive_topic"].lower() in text

    delta = 0
    if trigger_hits >= 3:
        delta += 2
    elif trigger_hits >= 1:
        delta += 1

    if sensitive_hit:
        delta += 1

    return delta


@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/api/new-game", methods=["POST"])
def new_game():
    data = _parse_json_body()
    session_id = data.get("session_id", str(random.randint(100000, 999999)))
    suspect = generate_suspect(groq_client=client)

    sessions[session_id] = {
        "suspect": suspect,
        "history": [],
        "start_time": time.time(),
        "message_count": 0,
        "accusation_made": False,
        "stress_level": 0,
        "hint_count": 0,
        "confessed": False,
        "last_analyzed": 0,
    }

    return jsonify({
        "session_id": session_id,
        "suspect_name": suspect["name"],
        "suspect_description": suspect["appearance"],
        "crime_category": suspect["crime_category"],
        "opening_statement": suspect["opening_statement"],
        "ascii_portrait": suspect.get("ascii_portrait", ""),
        "case_briefing": suspect.get("case_briefing", ""),
        "stress_level": 0,
    })


@app.route("/api/interrogate", methods=["POST"])
def interrogate():
    data = _parse_json_body()
    session_id = data.get("session_id")
    player_message = data.get("message", "").strip()

    if not session_id or session_id not in sessions:
        return jsonify({"error": "Session not found"}), 400
    if not player_message:
        return jsonify({"error": "Empty message"}), 400

    session = sessions[session_id]
    suspect = session["suspect"]
    stress = session["stress_level"]
    confessing = stress >= 10 and not session["confessed"]

    messages = [
        {"role": "system", "content": get_system_prompt(suspect, stress, confessing=confessing)}
    ]
    messages.extend(session["history"])
    messages.append({"role": "user", "content": player_message})

    def generate():
        # The frontend renders tokens live, but future turns need the completed
        # assistant message stored exactly once at the end of the stream.
        full_reply = []

        try:
            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=min(0.7 + stress * 0.03, 1.1),
                max_tokens=220,
                stream=True,
            )

            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    full_reply.append(token)
                    yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            return

        complete_reply = "".join(full_reply)
        session["history"].append({"role": "user", "content": player_message})
        session["history"].append({"role": "assistant", "content": complete_reply})
        session["message_count"] += 1

        if len(session["history"]) > 30:
            session["history"] = session["history"][-30:]

        delta = _calculate_stress_delta(player_message, complete_reply, suspect)
        session["stress_level"] = min(10, session["stress_level"] + delta)

        if confessing:
            session["confessed"] = True

        yield f"data: {json.dumps({'done': True, 'stress_level': session['stress_level'], 'confession': confessing, 'message_count': session['message_count']})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.route("/api/accuse", methods=["POST"])
def accuse():
    data = _parse_json_body()
    session_id = data.get("session_id")
    accusation = {
        "crime": data.get("crime", "").strip(),
        "motive": data.get("motive", "").strip(),
        "evidence": data.get("evidence", "").strip(),
    }

    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 400

    session = sessions[session_id]
    if session["accusation_made"]:
        return jsonify({"error": "Already accused"}), 400

    session["accusation_made"] = True
    suspect = session["suspect"]
    elapsed = time.time() - session["start_time"]

    result = evaluate_accusation(
        accusation,
        suspect,
        session["message_count"],
        elapsed,
        groq_client=client,
    )

    return jsonify({
        "result": result,
        "truth": {
            "crime": suspect["crime"],
            "motive": suspect["motive"],
            "secret": suspect["secret"],
            "key_contradiction": suspect["key_contradiction"],
        },
        "suspect_name": suspect["name"],
        "confessed": session.get("confessed", False),
    })


@app.route("/api/hint", methods=["POST"])
def hint():
    data = _parse_json_body()
    session_id = data.get("session_id")

    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 400

    session = sessions[session_id]
    suspect = session["suspect"]
    stress = session["stress_level"]
    session["hint_count"] += 1

    if stress >= 7:
        hints = [
            f"The suspect is cracking - push hard on {suspect['sensitive_topic']} right now.",
            f"Their alibi is unravelling. Ask them to repeat exactly what happened involving {suspect['hint_keyword']}.",
            "They have already contradicted themselves. Read back an earlier answer and demand they explain.",
        ]
    elif stress >= 4:
        hints = [
            f"The suspect reacted strangely to {suspect['hint_keyword']}.",
            f"They seem uncomfortable discussing {suspect['sensitive_topic']}.",
            "Ask about the timeline again - the numbers do not add up.",
        ]
    else:
        hints = [
            f"Try asking about {suspect['hint_keyword']}.",
            "Ask them to walk you through the exact timeline, hour by hour.",
            f"They seem uncomfortable discussing {suspect['sensitive_topic']}.",
        ]

    return jsonify({"hint": random.choice(hints), "stress_level": stress})


@app.route("/api/suggested-questions", methods=["POST"])
def suggested_questions():
    data = _parse_json_body()
    session_id = data.get("session_id")

    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 400

    suspect = sessions[session_id]["suspect"]

    prompt = f"""You are helping a player interrogate a suspect in a noir detective game.

Suspect public info:
- Name: {suspect['name']}
- Occupation: {suspect['occupation']}
- Crime category: {suspect['crime_category']}
- Their opening statement: {suspect['opening_statement']}

Generate exactly 5 short interrogation questions the player could ask. Each question should be a natural thing a detective would say - direct, probing, and varied in angle.

Return only a JSON array of 5 strings, with no markdown or explanation:
["question 1", "question 2", "question 3", "question 4", "question 5"]"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=300,
        )
        raw = _strip_markdown_fences(response.choices[0].message.content.strip())
        questions = json.loads(raw)
        if not isinstance(questions, list):
            raise ValueError("Expected a JSON list")
        questions = [str(question) for question in questions[:5]]
    except Exception as exc:
        print(f"[suggested_questions] failed: {exc}")
        questions = [
            "Where were you when it happened?",
            "How well did you know the victim?",
            "Walk me through your alibi again, exactly.",
            "That is not what you said before.",
            "Who can confirm your whereabouts?",
        ]

    return jsonify({"questions": questions})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = _parse_json_body()
    session_id = data.get("session_id")

    if session_id not in sessions:
        return jsonify({"contradiction": None}), 400

    session = sessions[session_id]
    if session["message_count"] < 6 or session["message_count"] - session["last_analyzed"] < 5:
        return jsonify({"contradiction": None})

    session["last_analyzed"] = session["message_count"]

    history = session["history"][-20:]
    transcript = "\n".join(
        f"{'Detective' if msg['role'] == 'user' else 'Suspect'}: {msg['content']}"
        for msg in history
    )

    prompt = f"""You are analyzing an interrogation transcript for inconsistencies.

Transcript:
{transcript}

Has the suspect contradicted themselves in any specific, concrete way? Look for:
- Different times or dates mentioned for the same event
- Claiming not to know something, then showing they know it
- Describing being in two places at overlapping times
- Any other specific factual inconsistency

If you find a real contradiction, respond with only this JSON:
{{"found": true, "note": "one sentence describing the specific contradiction"}}

If no clear contradiction exists yet, respond with only:
{{"found": false, "note": null}}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=120,
        )
        raw = _strip_markdown_fences(response.choices[0].message.content.strip())
        result = json.loads(raw)
        if result.get("found") and result.get("note"):
            return jsonify({"contradiction": result["note"]})
    except Exception as exc:
        print(f"[analyze] failed: {exc}")

    return jsonify({"contradiction": None})


@app.route("/api/stress", methods=["POST"])
def get_stress():
    data = _parse_json_body()
    session_id = data.get("session_id")

    if session_id not in sessions:
        return jsonify({"stress_level": 0})

    return jsonify({"stress_level": sessions[session_id]["stress_level"]})


if __name__ == "__main__":
    app.run(
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
    )
