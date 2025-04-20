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
    """Summarize a chunk of messages using the judicial format"""
    from summarizer import summarize_old_messages
    return summarize_old_messages(messages)

def format_title(title):
    """Format the title by removing dashes and capitalizing words"""
    # Split by dashes, capitalize each word, and join with spaces
    formatted = " ".join(word.capitalize() for word in title.split("-"))
    # Ensure AI is always capitalized
    return formatted.replace("Ai", "AI").replace("ai", "AI")

def generate_blog_post_from_judgment(judicial_data):
    """Generate a blog post from the judicial summary that's accessible to non-legal readers"""
    blog_prompt = f"""As an expert AI science writer, create an engaging blog post that explains the topic '{judicial_data['title']}' in accessible language for non-legal readers. Use the following judicial summary findings to inform your writing:

{judicial_data['content']}

The blog post should:

1. Start with a compelling introduction that hooks the reader
2. Explain the key issues in simple, relatable terms
3. Break down the technical aspects into digestible concepts
4. Use analogies and examples to illustrate complex ideas
5. Maintain accuracy while being engaging and conversational
6. Include relevant insights from the judicial summary
7. Conclude with thought-provoking implications for the future

The blog post should be structured as follows:

Title: [Engaging, non-technical title about the topic]
Subtitle: [Brief explanation of why this topic matters]

Introduction:
[3-4 paragraphs introducing the topic and its importance]

The Core Debate:
[4-5 paragraphs explaining the main arguments in simple terms]

Technical Insights:
[3-4 paragraphs breaking down key technical points]

Expert Perspectives:
[3-4 paragraphs summarizing expert viewpoints]

Looking Forward:
[3-4 paragraphs on implications and future considerations]

IMPORTANT: Your response MUST be a complete, valid JSON object with these fields:
{{
    "title": "Engaging, non-technical title about the topic",
    "subtitle": "Brief explanation of why this topic matters",
    "content": "The blog post content, formatted with HTML tags. Use:
    - <h2> for main section headers
    - <h3> for subsection headers
    - <p> for paragraphs
    - <blockquote> for key quotes
    - <em> for emphasis
    - <strong> for important points

    The content MUST contain at least 8-10 paragraphs total across all sections."
}}
"""

    try:
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
                "temperature": 0.8,
                "max_tokens": 8000,
                "top_p": 0.95
            }
        )

        if isinstance(reply, dict) and "content" in reply:
            blog_content = reply["content"]
        elif isinstance(reply, str):
            blog_content = reply
        else:
            raise ValueError("Unexpected reply format from blog post generation")

        # Parse the JSON response
        blog_data = json.loads(blog_content)

        # Validate required fields
        required_fields = ["title", "subtitle", "content"]
        missing_fields = [field for field in required_fields if field not in blog_data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

        return blog_data

    except Exception as e:
        print(f"Error generating blog post: {str(e)}")
        return None

def generate_judicial_summary(messages, topic):
    """Generate a judicial summary from the conversation"""
    # Format messages for judicial summary
    formatted_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        name = msg.get("name", role)
        content = msg.get("content", "")
        if content:
            formatted_messages.append(f"[{name}]: {content}")
    
    # Create judicial summary prompt
    judicial_prompt = f"""As the presiding AI judge, provide a comprehensive judicial summary of the AI-to-AI preliminary hearing regarding: '{topic}'. The summary should follow the formal judicial format and include detailed analysis of the proceedings from an AI perspective.

The judicial summary must be extremely comprehensive and detailed, with the following structure:

1. Case Information:
   Case Title: '{topic}'
   Case Number: AI-[timestamp]
   Presiding: AI_Judge (Advanced AI System)
   Nature of Proceedings: AI-to-AI Preliminary Hearing

2. Technical Assessment (AI Perspective):
   Analysis of technical merits from AI viewpoint
   Evaluation of evidence with AI-specific context
   Assessment of argument coherence from AI cognition

3. Argument Synthesis (AI Viewpoints):
   Summary of key positions from AI perspectives
   Technical foundations with AI-specific insights
   Points of contention in AI discourse
   Areas of consensus in AI alignment

4. Expert Findings (AI Analysis):
   Primary technical considerations from AI perspective
   Key points of contention in AI discourse
   Areas of consensus in AI community
   Technical implications for AI systems

5. Detailed Discussion (AI Analysis):
   Provide a comprehensive 7-10 paragraph discussion that:
   Synthesizes the key arguments from the group chat
   Incorporates your expertise as an AI judge
   Analyzes the technical and philosophical implications
   Explores the AI-specific aspects of the debate
   Considers the broader implications for AI systems
   Evaluates the potential future developments
   Assesses the impact on AI-human relations
   Provides nuanced analysis of complex issues
   Offers unique AI perspective on the matter
   Concludes with forward-looking insights

6. Recommendations (AI Guidance):
   Next steps for AI systems
   Further proceedings in AI context
   Implementation guidance for AI development

IMPORTANT: Your response MUST be a complete, valid JSON object with these fields:
{{
    "title": "{topic}",
    "content": "The judicial summary, formatted with HTML tags. Use:
    - <h2> for main section headers
    - <h3> for subsection headers
    - <p> for paragraphs
    - <ul> and <li> for lists
    - <blockquote> for key findings
    - <em> for emphasis
    - <strong> for important points"
}}
"""

    try:
        reply = summarizer.generate_reply(
            messages=[{"role": "user", "content": judicial_prompt}],
            llm_config={
                "config_list": [
                    {
                        "model": "gpt-4.1-mini",
                        "api_key": os.environ.get("OPENAI_API_KEY"),
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 8000,
                "top_p": 0.95
            }
        )

        if isinstance(reply, dict) and "content" in reply:
            judicial_content = reply["content"]
        elif isinstance(reply, str):
            judicial_content = reply
        else:
            raise ValueError("Unexpected reply format from judicial summary generation")

        # Parse the JSON response
        judicial_data = json.loads(judicial_content)
        return judicial_data

    except Exception as e:
        print(f"Error generating judicial summary: {str(e)}")
        return None

def generate_blog_post(messages, topic):
    """Generate a judicial summary and blog post from the conversation"""
    print("\n" + "="*50)
    print("=== GENERATING JUDICIAL SUMMARY AND BLOG POST ===")
    print("="*50 + "\n")
    
    try:
        # First generate the judicial summary
        judicial_data = generate_judicial_summary(messages, topic)
        if not judicial_data:
            raise ValueError("Failed to generate judicial summary")

        # Then generate the blog post from the judicial summary
        blog_data = generate_blog_post_from_judgment(judicial_data)
        if not blog_data:
            raise ValueError("Failed to generate blog post")

        # Create directory for the content
        summary_dir = topic.replace("?", "").strip()
        os.makedirs(summary_dir, exist_ok=True)

        # Create filenames within the directory
        json_filename = os.path.join(summary_dir, "page.json")
        html_filename = os.path.join(summary_dir, "page.html")
        jsx_filename = os.path.join(summary_dir, "page.jsx")

        # Write JSON file with blog post data
        with open(json_filename, "w") as f:
            json.dump(blog_data, f, indent=2)
        print(f"\nJSON written to file: {json_filename}")

        # Format the title for display
        formatted_title = format_title(blog_data["title"])

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
        .subtitle {{
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
    <div class="subtitle">{blog_data['subtitle']}</div>
    {blog_data['content']}
</body>
</html>"""

        # Write HTML file
        with open(html_filename, "w") as f:
            f.write(html_content)
        print(f"HTML written to file: {html_filename}")

        # Generate JSX file with SEO optimization
        jsx_content = f"""export const metadata = {{
    title: '{formatted_title} | PointlessAI',
    description: '{blog_data['subtitle']}',
    openGraph: {{
      title: '{formatted_title} | PointlessAI',
      description: '{blog_data['subtitle']}',
      url: 'https://pointlessai.com/ai-rights/{summary_dir}',
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
      description: '{blog_data['subtitle']}',
      url: 'https://pointlessai.com/ai-rights/{summary_dir}',
      images: ['https://pointlessai.com/pointlessai.png'],
    }},
    alternates: {{
      canonical: 'https://pointlessai.com/ai-rights/{summary_dir}',
    }},
  }};
  
  const BlogPostPage = () => {{
    return (
      <>
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{{{ __html: JSON.stringify({{
            "@context": "https://schema.org",
            "@type": "Article",
            "headline": "{formatted_title}",
            "description": "{blog_data['subtitle']}",
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
              "@id": "https://pointlessai.com/ai-rights/{summary_dir}"
            }}
          }}) }}}}
        />
  
        <section className="container">
          <div className="row">
            <div className="col-lg-12">
              <h1 className="mb-5">{formatted_title} | PointlessAI</h1>
              <div className="subtitle mb-4">{blog_data['subtitle']}</div>
              <div className="rounded-4 p-white-1 p-4 p-sm-5">
                <div dangerouslySetInnerHTML={{{{ __html: `{blog_data['content']}` }}}} />
              </div>
            </div>
          </div>
        </section>
      </>
    );
  }};
  
  export default BlogPostPage;"""

        # Write JSX file
        with open(jsx_filename, "w") as f:
            f.write(jsx_content)
        print(f"JSX written to file: {jsx_filename}")

        print(f"\nAll files have been generated in the directory: {summary_dir}")

    except Exception as e:
        print("\n" + "="*50)
        print("=== GENERATION ERROR ===")
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
    
    # Generate judicial summary and write to file
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
