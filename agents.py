import json
import re

from groq import Groq

from config import GROQ_API_KEY, MODEL


client = Groq(api_key=GROQ_API_KEY)


def parse_json(content):
    content = content.strip()
    content = re.sub(r"^```json\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"\s*```$", "", content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1:
            return json.loads(content[start:end + 1])
        raise ValueError("Invalid JSON returned by AI.")


def manager_agent(question):
    prompt = f"""
You are the manager agent of an enterprise AI knowledge assistant.

Classify the user's question into exactly one category:

HR
TECHNICAL
PROJECT
GENERAL

Question:

{question}

Return ONLY JSON:

{{
    "category": "HR"
}}

Do not provide an answer. Only classify the question.
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an intelligent routing agent."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    result = parse_json(response.choices[0].message.content)
    category = result.get("category", "GENERAL").upper()
    if category not in ["HR", "TECHNICAL", "PROJECT", "GENERAL"]:
        category = "GENERAL"
    return category


def generate_answer(question, context, agent_name):
    prompt = f"""
You are the {agent_name} in an enterprise knowledge assistant.

Answer the user's question using ONLY the provided company documentation.

Do not use outside knowledge.

If the documentation does not contain enough information, say:

"I could not find this information in the available company documents."

Do not invent policies, people, dates, procedures or technical information.

USER QUESTION:

{question}

DOCUMENT CONTEXT:

{context}

Provide:

1. Clear answer
2. Important details
3. Sources

Keep the answer professional and concise.
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a grounded enterprise AI assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content
