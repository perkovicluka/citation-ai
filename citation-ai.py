import os
import requests
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableConfig
from langgraph.graph import StateGraph
from langchain_openai import AzureChatOpenAI
from typing import TypedDict

# Define the state schema
class CitationState(TypedDict, total=False):
    sentence: str
    query: str
    source_title: str
    source_url: str
    quote: str
    enhanced: str

# Load environment variables
load_dotenv()

# Initialize AzureChatOpenAI
llm = AzureChatOpenAI(
    azure_endpoint="https://mavenaiservice.openai.azure.com/",
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2024-08-01-preview",
    azure_deployment="gpt-4o",
    temperature=0
)

# Node 1: Generate search query
def generate_query_node(state: dict) -> dict:
    sentence = state.get("sentence", "")
    prompt = f"Generate a concise search query to find a reliable source for the following claim:\n\n{sentence}"
    query = llm.invoke(prompt, config=RunnableConfig(run_name="generate_query")).content
    return {"query": query, "sentence": sentence}

# Node 2: Search for source using LangSearch
def search_source_node(state: dict) -> dict:
    query = state.get("query", "")
    headers = {
        "Authorization": f"Bearer {os.getenv('LANGSEARCH_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": query,
        "freshness": "noLimit",
        "summary": True,
        "count": 1
    }

    response = requests.post("https://api.langsearch.com/v1/web-search", headers=headers, json=payload)

    if response.status_code != 200:
        return {
            "source_title": "LangSearch API error",
            "source_url": "",
            "quote": "",
            "sentence": state.get("sentence", ""),
            "query": query
        }

    results = response.json()
    pages = results.get("data", {}).get("webPages", {}).get("value", [])
    if not pages:
        return {
            "source_title": "No sources found.",
            "source_url": "",
            "quote": "",
            "sentence": state.get("sentence", ""),
            "query": query
        }

    page = pages[0]
    title = page.get("name", "")
    url = page.get("url", "")
    quote = page.get("snippet", "") or page.get("summary", "")

    return {
        "source_title": title,
        "source_url": url,
        "quote": quote,
        "sentence": state.get("sentence", ""),
        "query": query
    }

# Node 3: Rewrite sentence with citation and quote
def rewrite_sentence_node(state: dict) -> dict:
    sentence = state.get("sentence", "")
    quote = state.get("quote", "")
    source_title = state.get("source_title", "")
    source_url = state.get("source_url", "")
    prompt = f"""You are an expert technical writer.

        Unless you believe otherwise, assume you are to provide a quote and citation to support a claim made in the sentence.

        1. Rewrite the following input sentence in a formal essay style using the provided quote.  

        2. If the input is a well-structured paragraph, do not completely rewrite it. Instead, insert or replace a section with the quote, transitioning smoothly.  
        
        3. Include an in-text citation in APA style with the quote in quotation marks "".  
            - Format: (Author, Year). If the author is mentioned in the sentence, only include the year in parentheses.  
            - Do not include direct links in the output paragraph.

        4. Match the sentence count of the input.  
            - If the input is a single sentence, the output should also be a single sentence.  
            - If the input is a multi-sentence paragraph, the output should contain the same number of sentences.  
            - You may slightly adjust sentence boundaries for flow, but do not add or remove entire sentences.  
            - The quote and citation must be integrated within this constraint.

        5. In the Sources section:  
            - Include all sources used in your report.  
            - Format each source in APA style: AuthorLastName, FirstInitial. (Year). *Title of the webpage or article*. Website Name. URL  
            - Separate each source by a newline. Use two spaces at the end of each line to create a newline in Markdown.  
            - Example:  Smith, J. (2023). *Understanding AI advancements*. OpenAI Blog. https://openai.com/blog/understanding-ai  

        6. Combine sources to avoid redundancy. For example, this is incorrect:
            [1] https://ai.meta.com/blog/meta-llama-3-1/
            [2] https://ai.meta.com/blog/meta-llama-3-1/

            It should be:
            [1] https://ai.meta.com/blog/meta-llama-3-1/

        7. Inputs to use are below. If any input is missing or empty, omit that part from your response.
            Sentence: {sentence}  
            Quote: {quote}  
            Source Title: {source_title}  
            Source URL: {source_url}
            """

    enhanced = llm.invoke(prompt, config=RunnableConfig(run_name="rewrite_sentence")).content
    return {"enhanced": enhanced}

# Build LangGraph flow
graph_builder = StateGraph(CitationState)
graph_builder.add_node("generate_query", generate_query_node)
graph_builder.add_node("search_source", search_source_node)
graph_builder.add_node("rewrite_sentence", rewrite_sentence_node)

graph_builder.set_entry_point("generate_query")
graph_builder.add_edge("generate_query", "search_source")
graph_builder.add_edge("search_source", "rewrite_sentence")

citation_graph = graph_builder.compile()

# Example usage
if __name__ == "__main__":
    input_state = {
        "sentence": "LangGraph simplifies agent design by letting developers visually connect memory, planning, and tool nodes. This modular setup makes complex workflows easier to build and debug. It's ideal for experimenting with different agent types quickly."
    }
    output = citation_graph.invoke(input_state)
    print(output.get("enhanced", "No output"))
