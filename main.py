from agents import create_agents, get_random_config
from summarizer import summarize_old_messages
from autogen import GroupChat, GroupChatManager, Agent, AssistantAgent
import os
import sys
from typing import Dict, Any

class CustomAssistantAgent(AssistantAgent):
    def generate_reply(self, messages=None, sender=None, config=None):
        # Generate new random config for this message
        current_config = get_random_config()
        
        # Print the metadata for this specific message
        """
        print("\n=== Response Metadata ===")
        print(f"Agent: {self.name}")
        print("\nSystem prompt:")
        print(self.system_message)
        print("\nLLM Config:")
        print(self.llm_config.temperature)
        print(self.llm_config.config_list[0].max_tokens)
        print(self.llm_config.config_list[0].top_p)
        print(self.llm_config.config_list[0].api_key)
        print("========================\n")
        """

        # Update the config for this message
        self.llm_config = {
            "config_list": self.llm_config["config_list"],
            **current_config
        }

        # Call the parent's generate_reply
        return super().generate_reply(messages=messages, sender=sender, config=config)

def print_metadata(metadata: Dict[str, Any]):
    print("\n=== Response Metadata ===")
    print(f"Temperature: {metadata.get('temperature', 'N/A')}")
    print(f"Max Tokens: {metadata.get('max_tokens', 'N/A')}")
    print(f"Top P: {metadata.get('top_p', 'N/A')}")
    print(f"Token Usage: {metadata.get('token_usage', 'N/A')}")
    print("========================\n")

def message_callback(recipient, messages, sender, config):
    # Get the last message
    last_message = messages[-1] if messages else None
    
    # Only process if this is an agent message (not a system or manager message)
    if last_message and last_message.get("role") == "assistant":
        # Generate new random config for this message
        current_config = get_random_config()
        
        # Print the metadata for this specific message
        print("\n=== Response Metadata ===")
        print(f"Agent: {sender.name}")
        print(f"Temperature: {current_config['temperature']}")
        print(f"Max Tokens: {current_config['max_tokens']}")
        print(f"Top P: {current_config['top_p']}")
        print("\nSystem prompt:")
        print(sender.system_message)
        print("\nLLM Config:")
        print(sender.llm_config)
        print("========================\n")

        # Update the sender's config for this message by creating a new config
        sender.llm_config = {
            "config_list": sender.llm_config["config_list"],
            **current_config
        }

    return False, None  # Important: return False to allow normal message processing

MAX_TURNS = 12
MAX_CONTEXT_TOKENS = 3000

def print_messages(messages):
    print("\n=== Group Chat Messages ===")
    for msg in messages:
        name = msg.get("name") or msg.get("role")
        content = msg.get("content", "").strip()
        if content:
            print(f"\n[{name}]: {content}\n")
    print("==========================\n")

def start_discussion(topic: str):
    agents = create_agents()
    # Replace regular AssistantAgents with our CustomAssistantAgent
    custom_agents = []
    for agent in agents:
        if isinstance(agent, AssistantAgent):
            custom_agent = CustomAssistantAgent(
                name=agent.name,
                system_message=agent.system_message,
                llm_config=agent.llm_config
            )
            custom_agents.append(custom_agent)
        else:
            custom_agents.append(agent)
    
    group = GroupChat(
        agents=custom_agents,
        messages=[],
        max_round=MAX_TURNS,
        speaker_selection_method="round_robin"  # Force round-robin selection
    )
    manager = GroupChatManager(
        groupchat=group, 
        llm_config={
            "config_list": [
                {
                    "model": "gpt-4o-mini",
                    "api_key": os.environ.get("OPENAI_API_KEY"),
                }
            ],
            **get_random_config()
        }
    )

    print(f"\n--- Starting Discussion: {topic} ---\n")
    
    # Initialize the conversation with the user's message
    custom_agents[0].initiate_chat(manager, message=topic)

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
    start_discussion("Democrats are better than Republicans.")
