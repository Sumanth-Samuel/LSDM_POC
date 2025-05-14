# utils/cost.py
import tiktoken
from utils.llm import build_prompt

PRICING = {
    "text-embedding-3-small": {"input": 0.02},     # per 1M tokens
    "gpt-4o": {"input": 2.5, "output": 10.0}        # per 1M tokens
}

def count_tokens(text, model="gpt-4o"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def estimate_cost(question, schema_snippets, sql_output):
    embed_tokens = count_tokens(question, model="text-embedding-3-small")
    embed_cost = (embed_tokens / 1_000_000) * PRICING["text-embedding-3-small"]["input"]

    prompt = build_prompt(question, schema_snippets)
    prompt_tokens = count_tokens(prompt, model="gpt-4o")
    output_tokens = count_tokens(sql_output, model="gpt-4o")

    gpt_input_cost = (prompt_tokens / 1_000_000) * PRICING["gpt-4o"]["input"]
    gpt_output_cost = (output_tokens / 1_000_000) * PRICING["gpt-4o"]["output"]

    total_cost = embed_cost + gpt_input_cost + gpt_output_cost

    return {
        "Embedding Tokens": embed_tokens,
        "Embedding Cost": embed_cost,
        "Prompt Tokens": prompt_tokens,
        "Output Tokens": output_tokens,
        "GPT-4o Cost": gpt_input_cost + gpt_output_cost,
        "Total Cost": total_cost
    }
