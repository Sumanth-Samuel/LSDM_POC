# utils/llm.py
import openai
import re

def build_prompt(question: str, schema_snippets: list) -> str:
    return f"""
### Task
Generate a SQL Server query for this question:

[QUESTION]
{question}
[/QUESTION]

Use only fields present in the schema below. Return ONLY SQL — no explanations.

### Relevant Schema
{chr(10).join(schema_snippets)}

[SQL]
"""

def clean_sql_output(raw_sql: str) -> str:
    raw_sql = re.sub(r"```sql|```", "", raw_sql, flags=re.IGNORECASE).strip()
    for k in ['Explanation', 'Summary', 'This query']:
        if k in raw_sql:
            raw_sql = raw_sql.split(k)[0]
    return raw_sql.strip()

def get_sql_from_llm(prompt: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()
