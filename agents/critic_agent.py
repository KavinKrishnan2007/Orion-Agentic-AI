import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

def evaluate(task, output):
    prompt = f"""You are a strict quality evaluator for AI agent outputs.

Task given to agent:
{task}

Agent output:
{output}

Evaluate the output on these criteria:
1. Relevance - does it directly address the task?
2. Accuracy - is the information correct and specific?
3. Completeness - is the response thorough?
4. Clarity - is it well structured and easy to understand?

Respond with ONLY a JSON object in this exact format, no other text:
{{"score": <integer 1-10>, "feedback": "<one sentence explaining the score and what to improve>"}}"""

    response = _groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.2
    )
    raw = response.choices[0].message.content.strip()
    try:
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        score = int(result.get("score", 5))
        feedback = result.get("feedback", "No feedback provided.")
        return score, feedback
    except Exception:
        return 5, f"Could not parse critic response: {raw[:100]}"
