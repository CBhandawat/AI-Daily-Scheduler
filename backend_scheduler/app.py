import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from google.generativeai import GenerativeModel, configure
import re

# Load .env variables
load_dotenv()

# Configure Gemini API
configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Gemini model (using Gemini 1.5 Pro or Pro Vision)
model = GenerativeModel("gemini-2.0-flash")

# Flask app
app = Flask(__name__)
CORS(app)

@app.route("/parse_prompt", methods=["POST"])
def parse_prompt():
    data = request.get_json()
    user_input = data.get("prompt", "")
    if not user_input:
        return jsonify({"error": "No prompt provided"}), 400

    prompt = (
        "You are an intelligent scheduling assistant.\n"
        "Given a user's plain text day plan, do the following:\n\n"
        "IMPORTANT RULES:\n"
        "- Do NOT make assumptions about the timing of tasks that are clearly unknown, such as meetings, appointments, going somewhere, or any specific activities the user mentioned without a time.\n"
        "- You CAN make reasonable default assumptions about the timing of meals (breakfast, lunch, dinner) because these words imply typical times.\n"
        "- If any time is missing or unclear, generate clarifying questions instead of assuming.\n\n"
        "Output a JSON object with:\n"
        "- 'parsed_tasks': an array of objects with 'task' and 'start' (if known)\n"
        "- 'clarifying_questions': an array of questions to ask the user\n\n"
        "Return ONLY the JSON object. No extra commentary.\n\n"
        "User Input:\n"
        + user_input
    )

    response = model.generate_content(prompt)

    reply = response.text.strip()
    print("--------- RAW GEMINI REPLY ---------")
    print(reply)
    print("-------------------------------------")

    # Remove Markdown wrapping if present
    if reply.startswith("```"):
        # Remove the opening ```
        reply = re.sub(r"^```json\s*", "", reply)
        # Remove the trailing ```
        reply = re.sub(r"\s*```$", "", reply)

    try:
        parsed = json.loads(reply)
        return jsonify(parsed)
    except Exception as e:
        return jsonify({"error": f"Invalid JSON from Gemini: {e}", "raw_reply": reply}), 500


@app.route("/generate_schedule", methods=["POST"])
def generate_schedule():
    """
    Generates the schedule JSON from prompt and clarifications.
    """
    data = request.get_json()
    prompt = data.get("prompt", "")
    clarifications = data.get("answers", [])

    schedule = generate_schedule_from_gemini(prompt, clarifications)
    return jsonify(schedule)


def generate_schedule_from_gemini(prompt, clarifications):
    """
    Calls Gemini to generate a structured schedule.
    """
    clarification_text = "\n".join(
        f"- {c['question']} Answer: {c['answer']}" for c in clarifications
    )


    final_prompt = f"""
You are an AI personal daily schedule planner.

RULES:
1. You MUST avoid asking any clarifying questions. 
2. If any information is missing, make a reasonable assumption and mention your assumption in the "note" field.
3. You MUST auto-schedule default tasks like drinking water (8 glasses evenly spaced) and meals at standard times (Breakfast ~8 AM, Lunch ~1 PM, Dinner ~7 PM).
4. If the schedule conflicts, adjust the times logically.

Return ONLY a JSON object with:
- "schedule": sorted list of tasks (each with "task", "start", "duration")
- "note": a string explaining any assumptions or adjustments you made

IMPORTANT:
- Do NOT include any clarifying_questions.
- Do NOT include any text outside JSON.

EXAMPLE RESPONSE:
{{
  "schedule": [
    {{"task": "Drink water (Glass 1)", "start": "08:00 AM", "duration": "5 minutes"}},
    {{"task": "Lunch", "start": "01:00 PM", "duration": "30 minutes"}}
  ],
  "note": "Assumed wake time as 7 AM since it was not provided."
}}

User Request:
{prompt}

Clarification Answers Provided by the User:
{clarification_text}
"""

    # Gemini call
    response = model.generate_content(final_prompt)

    json_text = response.text.strip()

    if not json_text:
        print("Gemini returned empty text.")
        raise ValueError("Gemini returned no output.")

    print("----- Gemini raw response -----")
    print(json_text)
    print("-------------------------------")

    # Remove triple backticks if present
    cleaned_text = json_text
    if json_text.startswith("```"):
        # Extract content between ```json and ```
        cleaned_text = re.sub(r"^```json\s*", "", json_text)
        cleaned_text = re.sub(r"\s*```$", "", cleaned_text)

    try:
        schedule = json.loads(cleaned_text)
        return schedule
    except json.JSONDecodeError as e:
        print("Failed to parse JSON from Gemini:")
        print(cleaned_text)
        raise ValueError(f"JSON parsing failed: {e}")

@app.route("/update_schedule", methods=["POST"])
def update_schedule():
    data = request.get_json()
    prompt = data.get("prompt", "")
    clarifications = data.get("clarifications", [])
    previous_schedule = data.get("previous_schedule", [])
    instructions = data.get("instructions", "")

    updated = generate_updated_schedule(prompt, clarifications, previous_schedule, instructions)
    return jsonify(updated)


def generate_updated_schedule(prompt, clarifications, previous_schedule, instructions):
    """
    Calls Gemini to revise the schedule based on additional instructions.
    """
    clar_text = "\n".join(
        f"- {c['question']} Answer: {c['answer']}" for c in clarifications
    )
    schedule_json = json.dumps(previous_schedule, indent=2)

    final_prompt = f"""
You are an AI daily schedule planner.

TASK:
Revise the existing schedule below based on the user's additional instructions.

RULES:
1. Keep all previous tasks unless the user explicitly says to remove them.
2. Adjust only what is specified.
3. Make reasonable adjustments if times conflict.
4. Do NOT ask any further clarifying questions.

Return ONLY a JSON object with:
- "schedule": list of tasks (each with "task", "start", "duration")
- "note": explanation of changes

EXISTING SCHEDULE:
{schedule_json}

Original User Request:
{prompt}

Clarifications:
{clar_text}

Additional Instructions:
{instructions}
"""

    response = model.generate_content(final_prompt)

    json_text = response.text.strip()

    if json_text.startswith("```"):
        json_text = re.sub(r"^```json\s*", "", json_text)
        json_text = re.sub(r"\s*```$", "", json_text)

    try:
        return json.loads(json_text)
    except Exception as e:
        print("Error parsing updated schedule:", json_text)
        raise ValueError(f"JSON parsing failed: {e}")

if __name__ == "__main__":
    app.run(debug=True)
