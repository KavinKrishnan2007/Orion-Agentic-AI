import os
import json
import uuid
from groq import Groq
from dotenv import load_dotenv
from agents import document_agent, api_caller_agent, critic_agent
from utils.cosmos_logger import log_event

load_dotenv()

_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

CRITIC_THRESHOLD = 6
MAX_RETRIES = 1

def decompose_goal(goal):
    prompt = f"""You are an AI orchestrator. Break down the following user goal into 2-3 specific sub-tasks.
For each sub-task assign a type from: "document", "api", "general"

- Use "document" for tasks involving uploaded PDF content, document Q&A, or text analysis
- Use "api" for tasks requiring real-time data: weather, stock prices, crypto prices  
- Use "general" for reasoning, summarization, or knowledge tasks

User goal: {goal}

Respond with ONLY a JSON array, no other text:
[
  {{"type": "document|api|general", "description": "specific task description"}},
  ...
]"""
    response = _groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.3
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    tasks = json.loads(raw)
    return tasks

def run_general_task(task_desc, workflow_id):
    response = _groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a knowledgeable assistant. Provide thorough, accurate answers."},
            {"role": "user", "content": task_desc}
        ],
        max_tokens=1024
    )
    answer = response.choices[0].message.content
    log_event(workflow_id, "GeneralAgent", task_desc, answer)
    return answer

def route_task(task, workflow_id):
    task_type = task.get("type", "general")
    task_desc = task.get("description", "")
    if task_type == "document":
        return document_agent.query(task_desc, workflow_id)
    elif task_type == "api":
        return api_caller_agent.run(task_desc, workflow_id)
    else:
        return run_general_task(task_desc, workflow_id)

def compose_final_response(goal, results):
    summaries = "\n\n".join([
        f"Task {i+1} ({r['type']}): {r['description']}\nResult: {r['output']}"
        for i, r in enumerate(results)
    ])
    prompt = f"""You are an AI assistant synthesizing results from multiple specialized agents.

Original user goal: {goal}

Agent results:
{summaries}

Write a clear, concise final response that directly addresses the user's goal by combining all agent results. 
Do not mention the agents or internal processes."""
    response = _groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024
    )
    return response.choices[0].message.content

def run(goal):
    workflow_id = str(uuid.uuid4())[:8]
    results = []
    try:
        tasks = decompose_goal(goal)
    except Exception as e:
        return {
            "workflow_id": workflow_id,
            "goal": goal,
            "error": f"Goal decomposition failed: {str(e)}",
            "results": [],
            "final_response": "Could not process goal."
        }
    for task in tasks:
        task_desc = task.get("description", "")
        task_type = task.get("type", "general")
        output = route_task(task, workflow_id)
        score, feedback = critic_agent.evaluate(task_desc, output)
        retry_output = None
        if score < CRITIC_THRESHOLD:
            retry_task = {**task, "description": f"{task_desc}\n\nPrevious attempt was insufficient. Critic feedback: {feedback}. Please improve."}
            retry_output = route_task(retry_task, workflow_id)
            retry_score, retry_feedback = critic_agent.evaluate(task_desc, retry_output)
            final_output = retry_output
            final_score = retry_score
        else:
            final_output = output
            final_score = score
        log_event(workflow_id, f"{task_type.capitalize()}Agent", task_desc, final_output, final_score)
        results.append({
            "type": task_type,
            "description": task_desc,
            "output": final_output,
            "critic_score": final_score,
            "critic_feedback": feedback,
            "retried": retry_output is not None
        })
    final_response = compose_final_response(goal, results)
    log_event(workflow_id, "Orchestrator", goal, final_response)
    return {
        "workflow_id": workflow_id,
        "goal": goal,
        "results": results,
        "final_response": final_response
    }
