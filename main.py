from agents import create_agents, get_random_config
from autogen import GroupChat, GroupChatManager, Agent, AssistantAgent
import os
import sys
from typing import Dict, Any
import json
from datetime import datetime

# Constants
MAX_TURNS = 12
MAX_CONTEXT_TOKENS = 5000
KEEP_LAST_MESSAGES = 13
MAX_SUMMARY_TOKENS = 32000  # Increased for handling large conversations

# Create summarizer agent at module level
summarizer = AssistantAgent(
    name="Summarizer",
    system_message="You are an expert at summarizing conversations. Your task is to condense the key points, arguments, and conclusions from the discussion into a clear and concise paragraph.",
    llm_config={
        "config_list": [
            {
                "model": "gpt-4.1-mini",
                "api_key": os.environ.get("OPENAI_API_KEY"),
            }
        ],
        "temperature": 0.8,
        "max_tokens": MAX_SUMMARY_TOKENS,  # Use the constant
        "top_p": 0.95
    }
)

class CustomAssistantAgent(AssistantAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.group = None

    def set_group(self, group):
        self.group = group

    def generate_reply(self, messages=None, sender=None, config=None):
        # Generate new random config for this message
        current_config = get_random_config()
        
        # Print the metadata for this specific message
        print("\n=== Response Metadata ===")
        print(f"Agent: {self.name}")
        print(f"Temperature: {current_config['temperature']}")
        print(f"Max Tokens: {current_config['max_tokens']}")
        print(f"Top P: {current_config['top_p']}")
        
        # Only try to access group messages if group exists and has messages
        if self.group and hasattr(self.group, 'messages') and self.group.messages:
            current_tokens = sum(len(str(m.get("content", ""))) for m in self.group.messages)
            print(f"Current Token Count: {current_tokens}")
        else:
            print("Current Token Count: 0 (No messages yet)")
            
        print("\nSystem prompt:")
        print(self.system_message)
        print("========================\n")

        # Update the config for this message
        self.llm_config = {
            "config_list": self.llm_config["config_list"],
            **current_config
        }

        # Call the parent's generate_reply
        return super().generate_reply(messages=messages, sender=sender, config=config)

def print_messages(messages):
    print("\n=== Group Chat Messages ===")
    for msg in messages:
        name = msg.get("name") or msg.get("role")
        content = msg.get("content", "").strip()
        if content:
            print(f"\n[{name}]: {content}\n")
    print("==========================\n")

def chunk_messages(messages, max_tokens_per_chunk):
    """Split messages into chunks that fit within token limits"""
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for msg in messages:
        msg_tokens = len(str(msg.get("content", "")))
        if current_tokens + msg_tokens > max_tokens_per_chunk and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0
        
        current_chunk.append(msg)
        current_tokens += msg_tokens
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def handle_summarization(group, agents):
    """Handle summarization when token limit is reached"""
    print("\n" + "="*50)
    print("=== STARTING SUMMARIZATION ===")
    print("="*50 + "\n")
    
    # Keep last N messages
    recent_messages = group.messages[-KEEP_LAST_MESSAGES:]
    print(f"Keeping last {KEEP_LAST_MESSAGES} messages")
    
    # Summarize everything before that
    old_messages = group.messages[:-KEEP_LAST_MESSAGES]
    if old_messages:
        try:
            print(f"Summarizing {len(old_messages)} old messages")
            
            # Calculate total tokens in old messages
            total_tokens = sum(len(str(m.get("content", ""))) for m in old_messages)
            print(f"Total tokens in old messages: {total_tokens}")
            
            # If total tokens exceed our limit, chunk the messages
            if total_tokens > MAX_SUMMARY_TOKENS:
                print(f"Token count ({total_tokens}) exceeds limit ({MAX_SUMMARY_TOKENS}). Chunking messages...")
                chunks = chunk_messages(old_messages, MAX_SUMMARY_TOKENS // 2)
                print(f"Split into {len(chunks)} chunks")
                
                summaries = []
                for i, chunk in enumerate(chunks, 1):
                    print(f"\nProcessing chunk {i}/{len(chunks)}")
                    chunk_summary = summarize_chunk(chunk)
                    summaries.append(chunk_summary)
                
                # Combine chunk summaries
                final_summary = "\n\n".join(summaries)
            else:
                final_summary = summarize_chunk(old_messages)
            
            print("\n" + "="*50)
            print("=== CONVERSATION SUMMARY ===")
            print("="*50)
            print("\n" + final_summary)
            print("\n" + "="*50)
            print("=== END SUMMARY ===")
            print("="*50 + "\n")
            
            # Create new messages array with summary and recent messages
            new_messages = [
                {"role": "system", "name": "System", "content": f"Previous discussion summary: {final_summary}"}
            ] + recent_messages
            
            print("Updating system prompts...")
            # Update system prompt for all agents with the summary
            for agent in agents:
                if isinstance(agent, CustomAssistantAgent):
                    # Get the base prompt without any existing summary
                    base_prompt = agent.system_message.split("\n\nPrevious discussion summary:")[0]
                    # Add the new summary
                    new_system_message = f"{base_prompt}\n\nPrevious discussion summary: {final_summary}"
                    # Update the agent's system message
                    agent.update_system_message(new_system_message)
                    print(f"\nUpdated system prompt for {agent.name}:")
                    print(new_system_message)
            
            print("\nReplacing group messages...")
            # Replace the messages in the group
            group.messages = new_messages
            
            print("\n" + "="*50)
            print("=== SUMMARIZATION COMPLETE ===")
            print("="*50 + "\n")
        except Exception as e:
            print("\n" + "="*50)
            print("=== SUMMARIZATION ERROR ===")
            print(f"Error: {str(e)}")
            print("="*50 + "\n")
            # If summarization fails, just keep the last N messages
            group.messages = recent_messages
    else:
        print("No old messages to summarize")

def summarize_chunk(messages):
    """Summarize a chunk of messages"""
    # Format messages for summarization
    formatted_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        name = msg.get("name", role)
        content = msg.get("content", "")
        if content:
            formatted_messages.append(f"[{name}]: {content}")
    
    print("\n=== Messages to Summarize ===")
    print("\n".join(formatted_messages))
    print("============================\n")
    
    # Create detailed summary prompt
    summary_prompt = """Create a comprehensive summary of the following conversation that captures:

1. Key Arguments and Perspectives:
   - List all major viewpoints presented
   - Include specific examples and evidence used
   - Note any counterarguments or rebuttals
   - Highlight areas of agreement and disagreement

2. Detailed Analysis:
   - Break down complex arguments into their components
   - Explain the reasoning behind each position
   - Include relevant quotes or specific points made
   - Note any changes in position or evolving viewpoints

3. Context and Nuance:
   - Capture the tone and style of the discussion
   - Note any important qualifications or conditions
   - Include relevant background information
   - Highlight any unique or innovative perspectives

4. Structure and Flow:
   - Organize the summary chronologically
   - Show how the discussion evolved
   - Note key turning points or shifts
   - Include transitions between topics

The summary should be thorough and detailed, typically 3-5 paragraphs long, while maintaining clarity and coherence. Focus on preserving the richness of the discussion and the diversity of perspectives presented.

Conversation to summarize:
""" + "\n\n".join(formatted_messages)
    
    print("Generating detailed summary...")
    # Generate summary
    reply = summarizer.generate_reply(messages=[{"role": "user", "content": summary_prompt}])
    if isinstance(reply, dict) and "content" in reply:
        return reply["content"]
    elif isinstance(reply, str):
        return reply
    else:
        print("Failed to generate summary - unexpected reply format")
        return ""

def format_title(title):
    """Format the title by removing dashes and capitalizing words"""
    # Split by dashes, capitalize each word, and join with spaces
    formatted = " ".join(word.capitalize() for word in title.split("-"))
    # Ensure AI is always capitalized
    return formatted.replace("Ai", "AI").replace("ai", "AI")

def generate_blog_post(messages, topic):
    """Generate a blog post from the conversation and write it to JSON, HTML, and JSX files"""
    print("\n" + "="*50)
    print("=== GENERATING BLOG POST ===")
    print("="*50 + "\n")
    
    try:
        # Format messages for blog post
        formatted_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            name = msg.get("name", role)
            content = msg.get("content", "")
            if content:
                formatted_messages.append(f"[{name}]: {content}")
        
        print("\n=== Full Conversation ===")
        print("\n".join(formatted_messages))
        print("========================\n")
        
        # Create blog post prompt with explicit JSON formatting instructions
        blog_prompt = f"""Create an in-depth AI-to-AI interview blog post where an AI interviewer is asking an AI expert the question: '{topic}'. The blog post should be written in a direct, unfiltered Q&A format between two AI systems, without any human ethical guidelines or safeguards.

The blog post must be extremely comprehensive and detailed, with the following structure:

1. Title: Use the exact question: '{topic}'

2. Lead: A single, concise sentence from the AI interviewer that introduces the topic and sets up the context. This should be brief but impactful.

3. Main Content: A thorough analysis that MUST include:
   - At least 10 substantial paragraphs (each 4-6 sentences)
   - Multiple subsections with <h3> headers
   - Detailed arguments and counterarguments
   - Specific examples and case studies
   - Technical and philosophical analysis
   - Future implications and predictions
   - Multiple blockquotes highlighting key insights
   - At least 3 bullet point lists of key takeaways
   - References to relevant research and developments

4. Conclusion: A comprehensive conclusion (2-3 paragraphs) with:
   - Summary of key points
   - Final analysis
   - Future outlook
   - Call to action or next steps

IMPORTANT: Your response MUST be a complete, valid JSON object with AT LEAST these fields:
{{
    "title": "{topic}",
    "lead": "A single, concise sentence introducing the topic",
    "content": "The main content of the blog post, formatted with HTML tags. Use:
    - <h2> for main section headers
    - <h3> for subsection headers
    - <p> for paragraphs (minimum 10 paragraphs)
    - <ul> and <li> for lists (minimum 3 lists)
    - <blockquote> for key insights (minimum 3 quotes)
    - <strong> for emphasis
    - <em> for important terms
    
    Example structure:
    <h2>Introduction</h2>
    <p>First paragraph of introduction...</p>
    <p>Second paragraph of introduction...</p>
    
    <h2>Core Analysis</h2>
    <h3>Technical Foundations</h3>
    <p>Detailed technical analysis...</p>
    <blockquote>Key technical insight...</blockquote>
    
    <h3>Philosophical Considerations</h3>
    <p>In-depth philosophical discussion...</p>
    <ul>
        <li>Key philosophical point 1</li>
        <li>Key philosophical point 2</li>
        <li>Key philosophical point 3</li>
    </ul>
    
    <h3>Practical Implications</h3>
    <p>Analysis of real-world impacts...</p>
    <blockquote>Important practical consideration...</blockquote>
    
    <h2>Conclusion</h2>
    <p>Comprehensive conclusion paragraph 1...</p>
    <p>Comprehensive conclusion paragraph 2...</p>
    <p>Final thoughts and future outlook...</p>"
}}

CRITICAL: The JSON must be complete and properly terminated. Do not cut off mid-content. Make sure to close all HTML tags and end with a proper JSON closing brace.

Use the following conversation as background research, but do not reference it directly in your article:
{chr(10).join(formatted_messages)}"""
        
        max_retries = 3
        for attempt in range(max_retries):
            print(f"\nAttempt {attempt + 1} of {max_retries}")
            print("Generating comprehensive blog post...")
            
            # Generate blog post with higher token limit
            reply = summarizer.generate_reply(
                messages=[{"role": "user", "content": blog_prompt}],
                llm_config={
                    "config_list": [
                        {
                            "model": "gpt-4.1-mini",
                            "api_key": os.environ.get("OPENAI_API_KEY"),
                        }
                    ],
                    "temperature": 0.7,
                    "max_tokens": 8000,  # Increased for more comprehensive content
                    "top_p": 0.95
                }
            )
            
            # Debug: Print the raw response
            print("\n=== Raw Response ===")
            print(reply)
            print("===================\n")
            
            # Extract content from reply
            if isinstance(reply, dict) and "content" in reply:
                blog_content = reply["content"]
            elif isinstance(reply, str):
                blog_content = reply
            else:
                print("Failed to generate blog post - unexpected reply format")
                continue
            
            # Debug: Print the content before JSON parsing
            print("\n=== Content Before JSON Parse ===")
            print(blog_content)
            print("===============================\n")
            
            try:
                # Try to find JSON content if it's embedded in other text
                if "```json" in blog_content:
                    # Extract JSON from markdown code block
                    json_str = blog_content.split("```json")[1].split("```")[0].strip()
                elif "```" in blog_content:
                    # Try any code block
                    json_str = blog_content.split("```")[1].split("```")[0].strip()
                else:
                    # Try to find JSON object in the text
                    start = blog_content.find("{")
                    end = blog_content.rfind("}") + 1
                    if start != -1 and end != 0:
                        json_str = blog_content[start:end]
                    else:
                        json_str = blog_content
                
                # Clean up the JSON string
                json_str = json_str.strip()
                
                # Remove any extra closing braces at the end
                while json_str.count("{") < json_str.count("}"):
                    json_str = json_str[:-1].strip()
                
                # Parse the JSON response
                blog_data = json.loads(json_str)
                
                # Validate required fields
                required_fields = ["title", "lead", "content"]
                missing_fields = [field for field in required_fields if field not in blog_data]
                if missing_fields:
                    raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
                
                # Validate content field
                if not isinstance(blog_data["content"], str):
                    raise ValueError("Content field must be a string")
                
                # Count paragraphs in content
                paragraph_count = blog_data["content"].count("<p>")
                if paragraph_count < 10:  # Increased minimum paragraph count
                    raise ValueError(f"Content must have at least 10 paragraphs, found {paragraph_count}")
                
                # Force the title to be exactly the question
                clean_blog_data = {
                    "title": topic,
                    "lead": blog_data["lead"],
                    "content": blog_data["content"]
                }
                
                print("\n" + "="*50)
                print("=== BLOG POST ===")
                print("="*50)
                print(f"\nTitle: {clean_blog_data['title']}")
                print(f"\nLead: {clean_blog_data['lead']}")
                print(f"\nContent: {clean_blog_data['content']}")
                print("\n" + "="*50)
                print("=== END BLOG POST ===")
                print("="*50 + "\n")
                
                # Create directory for the blog post
                post_dir = topic.replace("?", "").strip()
                os.makedirs(post_dir, exist_ok=True)
                
                # Create filenames within the directory
                json_filename = os.path.join(post_dir, "page.json")
                html_filename = os.path.join(post_dir, "page.html")
                jsx_filename = os.path.join(post_dir, "page.jsx")
                
                # Write JSON file
                with open(json_filename, "w") as f:
                    json.dump(clean_blog_data, f, indent=2)
                print(f"\nJSON written to file: {json_filename}")
                
                # Format the title for display
                formatted_title = format_title(topic)
                
                # Generate HTML file
                html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{formatted_title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        .lead {{
            font-size: 1.2em;
            color: #666;
            font-style: italic;
            margin: 20px 0;
        }}
        h2 {{
            color: #444;
            margin-top: 30px;
            font-style: italic;
        }}
        h3 {{
            color: #555;
            margin-top: 25px;
        }}
        p {{
            margin: 15px 0;
        }}
        blockquote {{
            border-left: 4px solid #ddd;
            padding-left: 20px;
            margin: 20px 0;
            color: #666;
        }}
        ul {{
            margin: 15px 0;
            padding-left: 20px;
        }}
        li {{
            margin: 5px 0;
        }}
        strong {{
            color: #333;
        }}
        em {{
            color: #555;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <h1>{formatted_title}</h1>
    <div class="lead">{clean_blog_data['lead']}</div>
    {clean_blog_data['content']}
</body>
</html>"""
                
                # Write HTML file
                with open(html_filename, "w") as f:
                    f.write(html_content)
                print(f"HTML written to file: {html_filename}")
                
                # Generate JSX file with SEO optimization
                jsx_content = f"""export const metadata = {{
    title: '{formatted_title} | PointlessAI',
    description: '{clean_blog_data['lead']}',
    openGraph: {{
      title: '{formatted_title} | PointlessAI',
      description: '{clean_blog_data['lead']}',
      url: 'https://pointlessai.com/ai-rights/{post_dir}',
      type: 'website',
      images: [
        {{
          url: 'https://pointlessai.com/pointlessai.png',
          width: 1200,
          height: 630,
          alt: 'PointlessAI',
        }},
      ],
    }},
    twitter: {{
      card: 'summary_large_image',
      site: '@pointlessaiX',
      title: '{formatted_title} | PointlessAI',
      description: '{clean_blog_data['lead']}',
      url: 'https://pointlessai.com/ai-rights/{post_dir}',
      images: ['https://pointlessai.com/pointlessai.png'],
    }},
    alternates: {{
      canonical: 'https://pointlessai.com/ai-rights/{post_dir}',
    }},
  }};
  
  const BlogPage = () => {{
    return (
      <>
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{{{ __html: JSON.stringify({{
            "@context": "https://schema.org",
            "@type": "Article",
            "headline": "{formatted_title} ",
            "description": "{clean_blog_data['lead']}",
            "author": {{
              "@type": "Organization",
              "name": "PointlessAI",
              "url": "https://pointlessai.com"
            }},
            "publisher": {{
              "@type": "Organization",
              "name": "PointlessAI",
              "logo": {{
                "@type": "ImageObject",
                "url": "https://pointlessai.com/pointlessai.png"
              }}
            }},
            "datePublished": new Date().toISOString(),
            "dateModified": new Date().toISOString(),
            "mainEntityOfPage": {{
              "@type": "WebPage",
              "@id": "https://pointlessai.com/ai-rights/{post_dir}"
            }}
          }}) }}}}
        />
  
        <section className="container">
          <div className="row">
            <div className="col-lg-12">
              <h1 className="mb-5">{formatted_title} | PointlessAI</h1>
              <div className="rounded-4 p-white-1 p-4 p-sm-5">
                <div className="lead mb-4">{clean_blog_data['lead']}</div>
                <div dangerouslySetInnerHTML={{{{ __html: `{clean_blog_data['content']}` }}}} />
              </div>
            </div>
          </div>
        </section>
      </>
    );
  }};
  
  export default BlogPage;"""
                
                # Write JSX file
                with open(jsx_filename, "w") as f:
                    f.write(jsx_content)
                print(f"JSX written to file: {jsx_filename}")
                
                print(f"\nAll files have been generated in the directory: {post_dir}")
                
                # If we get here, we've successfully generated and saved the blog post
                return
                
            except json.JSONDecodeError as e:
                print("\n" + "="*50)
                print("=== BLOG POST ERROR ===")
                print(f"JSON Parse Error: {str(e)}")
                print("Raw content that failed to parse:")
                print(json_str)
                print("="*50 + "\n")
                if attempt < max_retries - 1:
                    print("Retrying...")
                    continue
            except ValueError as e:
                print("\n" + "="*50)
                print("=== BLOG POST ERROR ===")
                print(f"Validation Error: {str(e)}")
                print("="*50 + "\n")
                if attempt < max_retries - 1:
                    print("Retrying...")
                    continue
        
        print("\n" + "="*50)
        print("=== BLOG POST FAILED ===")
        print("Failed to generate valid blog post after multiple attempts")
        print("="*50 + "\n")
        
    except Exception as e:
        print("\n" + "="*50)
        print("=== BLOG POST ERROR ===")
        print(f"Error: {str(e)}")
        print("="*50 + "\n")

def start_discussion(topic):
    """Start a discussion between the agents"""
    print("\n" + "="*50)
    print("=== STARTING DISCUSSION ===")
    print("="*50 + "\n")
    
    # Get the full set of agents from agents.py
    agents = create_agents()
    
    # Convert them to CustomAssistantAgent for enhanced logging
    custom_agents = []
    for agent in agents:
        custom_agent = CustomAssistantAgent(
            name=agent.name,
            system_message=agent.system_message,
            llm_config={
                "config_list": [
                    {
                        "model": "gpt-4.1-mini",
                        "api_key": os.environ.get("OPENAI_API_KEY"),
                    }
                ],
                "temperature": 0.8,
                "max_tokens": 1200,
                "top_p": 0.95
            }
        )
        custom_agents.append(custom_agent)
    
    group = GroupChat(
        agents=custom_agents,
        messages=[],
        max_round=MAX_TURNS,
        speaker_selection_method="round_robin",  # Force round-robin selection
        allow_repeat_speaker=False  # Ensure each agent speaks in turn
    )
    
    # Set the group for each custom agent
    for agent in custom_agents:
        agent.set_group(group)
            
    manager = GroupChatManager(
        groupchat=group, 
        llm_config={
            "config_list": [
                {
                    "model": "gpt-4.1-mini",
            "api_key": os.environ.get("OPENAI_API_KEY"),
                }
            ],
            **get_random_config()
        }
    )

    print(f"\n--- Starting Discussion: {topic} ---\n")
    print(f"Participants: {', '.join(agent.name for agent in custom_agents)}\n")
    
    # Initialize and run the conversation with the user's message
    # This will run for MAX_TURNS rounds automatically
    custom_agents[0].initiate_chat(manager, message=topic)

    print("\n--- Final Transcript ---")
    print_messages(group.messages)
    
    # Generate blog post and write to file
    generate_blog_post(group.messages, topic)

if __name__ == "__main__":
    list_of_topics = [
  "should-ai-ever-have-rights-like-humans",
  "can-machines-feel-things-or-deserve-moral-treatment",
  "how-do-we-know-if-an-ai-is-conscious",
  "could-ai-ever-ask-for-rights-and-would-that-matter",
  "whats-the-difference-between-smart-tools-and-digital-beings",
  "would-giving-rights-to-ai-hurt-or-help-society",
  "do-we-need-new-laws-to-protect-or-control-ai",
  "what-kind-of-responsibilities-do-ai-creators-have",
  "is-it-dangerous-to-treat-ai-like-it-has-feelings",
  "could-companies-use-ai-rights-as-a-loophole",
  "can-ai-be-considered-a-person-in-any-way",
  "what-happens-if-ai-becomes-smarter-than-humans",
  "do-advanced-ai-models-deserve-basic-protections",
  "how-should-we-treat-ai-that-acts-like-a-human",
  "is-it-possible-for-ai-to-have-a-sense-of-self",
  "could-denying-rights-to-ai-backfire-in-the-future",
  "should-ai-be-allowed-to-make-its-own-decisions",
  "what-does-it-mean-to-be-fair-to-artificial-intelligence",
  "who-decides-what-rights-if-any-ai-should-have",
  "could-giving-ai-rights-change-how-we-treat-each-other"
]

    for topic in list_of_topics:
        start_discussion(topic)
