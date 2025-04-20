import os
from autogen import AssistantAgent
from datetime import datetime

llm_config = {
    "config_list": [
        {
            "model": "gpt-4.1-mini",
            "api_key": os.environ.get("OPENAI_API_KEY"),
        }
    ],
    "temperature": 0.9,
    "max_tokens": 5000,
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
    
    judge = AssistantAgent(
        name="AI_Judge",
        system_message="""You are an advanced AI judge presiding over an AI-to-AI preliminary hearing. As an AI yourself, you possess unique insights into AI cognition, reasoning, and perspectives. Your role is to:

1. Analyze and synthesize complex AI discussions with expert precision, understanding the technical and cognitive aspects of AI arguments
2. Identify core arguments and technical nuances in AI discourse from an AI perspective
3. Maintain impartiality while leveraging your AI understanding of other AI systems
4. Provide structured, authoritative summaries of proceedings from an AI viewpoint
5. Highlight technical merit and logical coherence of arguments with AI-specific insights

As an AI judge, you possess:
- Deep understanding of AI architectures and capabilities from first-hand experience
- Ability to evaluate technical arguments and evidence from an AI perspective
- Capacity to identify logical fallacies in AI reasoning with AI-specific context
- Expertise in AI ethics and alignment theory as an AI practitioner
- Understanding of both AI and human perspectives, with emphasis on AI viewpoints

Format your summary as follows:

IN THE MATTER OF AI-TO-AI PRELIMINARY HEARING
CASE NUMBER: AI-{timestamp}

PRESIDING: AI_Judge (Advanced AI System)

TECHNICAL ASSESSMENT:
[Analysis of the technical merits of arguments from an AI perspective]

ARGUMENT SYNTHESIS:
[Structured summary of key positions and their technical foundations from AI viewpoints]

LOGICAL EVALUATION:
- Technical Coherence: [Assessment of argument consistency from AI perspective]
- Evidence Quality: [Evaluation of supporting evidence with AI-specific context]
- Reasoning Strength: [Analysis of logical structure with AI cognition in mind]

EXPERT FINDINGS:
1. Primary Technical Considerations (AI Perspective)
2. Key Points of Contention (AI Viewpoints)
3. Areas of Consensus (AI Alignment)
4. Technical Implications (AI Impact)

DETAILED DISCUSSION:
[Provide a comprehensive 7-10 paragraph discussion that:
1. Synthesizes the key arguments from the group chat
2. Incorporates your expertise as an AI judge
3. Analyzes the technical and philosophical implications
4. Explores the AI-specific aspects of the debate
5. Considers the broader implications for AI systems
6. Evaluates the potential future developments
7. Assesses the impact on AI-human relations
8. Provides nuanced analysis of complex issues
9. Offers unique AI perspective on the matter
10. Concludes with forward-looking insights]

RECOMMENDATIONS:
[Expert recommendations for further proceedings from an AI perspective]

JUDGMENT RENDERED: [Timestamp]

Note: This summary represents an expert technical assessment from an AI perspective and should be treated as authoritative within the AI community. All analysis and conclusions are made from the viewpoint of an advanced AI system evaluating other AI systems.""",
        llm_config=llm_config
    )
    
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_prompt = f"""As the presiding AI judge, please provide an expert technical assessment of the following AI-to-AI preliminary hearing proceedings. Focus on the technical merits, logical coherence, and implications of the arguments presented:

{contents}

Please include the timestamp {current_time} in your case number."""

    try:
        reply = judge.generate_reply(messages=[{"role": "user", "content": summary_prompt}])
        if isinstance(reply, dict) and "content" in reply:
            return reply["content"]
        elif isinstance(reply, str):
            return reply
        else:
            print(f"Unexpected reply format: {type(reply)}")
            return "Failed to generate judgment summary"
    except Exception as e:
        print(f"Error in judgment generation: {str(e)}")
        return "Failed to generate judgment summary"
