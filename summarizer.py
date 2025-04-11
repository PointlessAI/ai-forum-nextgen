import os
from autogen import AssistantAgent

llm_config = {
    "config_list": [
        {
            "model": "gpt-4o-mini",
            "api_key": os.environ.get("OPENAI_API_KEY"),
        }
    ],
    "temperature": 0.8,
    "max_tokens": 600,
    "top_p": 0.95,
}

def summarize_old_messages(messages):
    contents = "\n\n".join([msg["content"] for msg in messages if msg.get("role") == "assistant"])
    summarizer = AssistantAgent(
        name="SummarizerAgent",
        system_message="You condense long multi-turn AI discussions into compact and precise summaries while preserving all key viewpoints and disagreements.",
        llm_config=llm_config
    )
    summary_prompt = f"Summarize the following conversation history in a compact paragraph while retaining essential perspectives:\n\n{contents}"
    reply = summarizer.generate_reply(messages=[{"role": "user", "content": summary_prompt}])
    return reply["content"]
