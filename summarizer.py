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
    "max_tokens": 1400,
    "top_p": 0.95,
}

def summarize_old_messages(messages):
    # Format messages for summarization, including role information
    formatted_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        name = msg.get("name", role)
        content = msg.get("content", "")
        if content:
            formatted_messages.append(f"[{name}]: {content}")
    
    contents = "\n\n".join(formatted_messages)
    
    summarizer = AssistantAgent(
        name="SummarizerAgent",
        system_message="You condense long multi-turn AI discussions into compact and precise summaries while preserving all key viewpoints and disagreements. Include the main arguments, counter-arguments, and any significant points made by different participants. Your response should be a single paragraph.",
        llm_config=llm_config
    )
    
    summary_prompt = f"""Summarize the following conversation history in a compact paragraph while retaining essential perspectives and viewpoints. Focus on the main arguments and key points made by different participants:

{contents}"""
    
    try:
        reply = summarizer.generate_reply(messages=[{"role": "user", "content": summary_prompt}])
        if isinstance(reply, dict) and "content" in reply:
            return reply["content"]
        elif isinstance(reply, str):
            return reply
        else:
            print(f"Unexpected reply format: {type(reply)}")
            return "Failed to generate summary"
    except Exception as e:
        print(f"Error in summarization: {str(e)}")
        return "Failed to generate summary"
