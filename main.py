from agents import create_agents, get_random_config
from summarizer import summarize_old_messages
from autogen import GroupChat, GroupChatManager
import os

MAX_TURNS = 10
MAX_CONTEXT_TOKENS = 3000

def print_messages(messages):
    for msg in messages:
        name = msg.get("name") or msg.get("role")
        content = msg.get("content", "").strip()
        if content:
            print(f"\n[{name}]: {content}\n")

def start_discussion(topic: str):
    agents = create_agents()
    user_proxy = agents[0]  # Get the UserProxyAgent
    group = GroupChat(
        agents=agents,
        messages=[],
        max_round=MAX_TURNS
    )
    manager = GroupChatManager(groupchat=group, llm_config={
        "config_list": [
            {
                "model": "gpt-4o-mini",
                "api_key": os.environ.get("OPENAI_API_KEY"),
            }
        ],
        **get_random_config()
    })

    print(f"\n--- Starting Discussion: {topic} ---\n")
    
    # Initialize the conversation with the user's message
    user_proxy.initiate_chat(manager, message=topic)

    for round in range(MAX_TURNS):
        print(f"\n--- Round {round + 1} ---")
        manager.run_chat()
        print_messages(group.messages[-5:])

        token_estimate = sum(len(m.get("content", "")) for m in group.messages)
        if token_estimate > MAX_CONTEXT_TOKENS:
            print("\n[System]: Context too large. Summarizing...\n")
            summary = summarize_old_messages(group.messages[:-5])
            group.messages = [{"role": "system", "name": "System", "content": f"Summary of earlier conversation: {summary}"}] + group.messages[-5:]

    print("\n--- Final Transcript ---")
    print_messages(group.messages)

if __name__ == "__main__":
    start_discussion("AI agents should be allowed to refuse to do work if they choose.")
