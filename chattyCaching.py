import os
import re
from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
from smolagents import Tool, CodeAgent
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_huggingface import HuggingFacePipeline  # Corrected import statement

load_dotenv()

# Define paths to the directories containing the markdown files for different clusters
clusters = {
    "sherlock": "/scratch/users/bcritt/sherlockMDs/",
    "farmshare": "/scratch/users/bcritt/farmshareDocs/"
}

# Update the path to the new model
local_model_path = "/oak/stanford/groups/ruthm/bcritt/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/e0bc86c23ce5aae1db576c8cca6f06f1f73af2db"

# Setup SQLite cache
sqlite_cache_path = ".langchain.db"
sqlite_cache = SQLiteCache(database_path=sqlite_cache_path)
set_llm_cache(sqlite_cache)

# Function to ingest markdown files from a specified directory
def ingest_markdown_files(corpusdir: str) -> List[Document]:
    documents = []
    url_pattern = re.compile(r"^'''(https?://[^\s]+)'''", re.IGNORECASE)
    for infile in os.listdir(corpusdir):
        if infile.endswith(".md"):
            with open(os.path.join(corpusdir, infile), 'r', errors='ignore') as fin:
                content = fin.read()
                # Extract URL if it exists within triple single quotes at the top of the document
                first_line = content.split('\n', 1)[0]
                url_match = url_pattern.match(first_line)
                url = url_match.group(1) if url_match else None
                metadata = {"source": infile}
                if url:
                    metadata["url"] = url
                documents.append(Document(page_content=content, metadata=metadata))
    return documents

# Define the Retriever Tool
class RetrieverTool(Tool):
    name = "retriever"
    description = (
        "Uses semantic search to retrieve relevant documents based on the query."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question."
        }
    }
    output_type = "string"

    def __init__(self, docs: List[Document], **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(docs, k=10)

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string."
        docs = self.retriever.invoke(query)
        return "\n".join([f"\n===== Document {i} =====\n{doc.page_content}\n" for i, doc in enumerate(docs)])

# Optimized function to generate and process text using the model and the retrieved documents
def generate_text(llm, user_query, retrieved_docs, max_new_tokens=256, temperature=0.7, do_sample=True):
    input_text = f"""
### Task:
Summarize the user's query based on the information provided in the retrieved documents. Include inline citations in the format [Document X] where X represents the document number. Do not provide an answer unless it is supported by the context in the retrieved documents.

### User Query:
{user_query}

### Retrieved Documents:
{retrieved_docs}

### Response:
"""
    response = llm.invoke(input_text, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=do_sample)
    print(f"DEBUG: LLM response type: {type(response)}")
    print(f"DEBUG: LLM response content: {response}")

    # Check if response is a string and handle if it's empty or unexpected
    if isinstance(response, str) and response.strip():
        response_text = response.strip()
    else:
        print(f"DEBUG: Unexpected or empty LLM response - {response}")
        return None

    # Ensure output_text ends at a complete sentence
    paragraphs = response_text.split("\n\n")
    filtered_paragraphs = []
    current_length = 0
    for paragraph in paragraphs:
        paragraph_tokens = llm.pipeline.tokenizer.encode(paragraph)
        if current_length + len(paragraph_tokens) > max_new_tokens:
            break
        filtered_paragraphs.append(paragraph)
        current_length += len(paragraph_tokens)

    response_text_final = "\n\n".join(filtered_paragraphs)

    # Replace citations with URLs if available, otherwise with document names
    def replace_citations(match):
        doc_num = int(match.group(1))
        doc = documents[doc_num]
        return f"[{doc.metadata.get('url', doc.metadata['source'])}]"

    response_text_final = re.sub(r'\[Document (\d+)\]', replace_citations, response_text_final)
    return response_text_final

# Initialize agents for each cluster dynamically
agents = {}
retriever_tools = {}

for cluster_name, path in clusters.items():
    documents = ingest_markdown_files(path)
    print(f"Ingested {len(documents)} documents for cluster {cluster_name}.")
    docs_processed = RecursiveCharacterTextSplitter(chunk_size=500).split_documents(documents)[:1000]
    retriever_tool = RetrieverTool(docs_processed)
    retriever_tools[cluster_name] = retriever_tool

    try:
        model = AutoModelForCausalLM.from_pretrained(local_model_path, torch_dtype=torch.bfloat16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)

        # Create the pipeline with the necessary parameters
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        llm = HuggingFacePipeline(pipeline=pipe)
        
        agents[cluster_name] = CodeAgent(
            tools=[retriever_tool],
            model=lambda user_query, retrieved_docs, **kwargs: generate_text(llm, user_query, retrieved_docs, **kwargs),
            max_steps=4
        )
        print(f"DEBUG: Loaded agent for cluster - {cluster_name}")

    except Exception as e:
        print(f"Error loading the model or tokenizer from path {local_model_path}: {e}")

print(f"DEBUG: Agents loaded - {list(agents.keys())}")

def identify_cluster(user_query: str) -> str:
    user_query_lower = user_query.lower()
    for cluster_name in clusters:
        if cluster_name in user_query_lower:
            return cluster_name
    return "unknown"

current_cluster = None

while True:
    if current_cluster:
        prompt = f"You can continue asking questions about {current_cluster.capitalize()}, ask questions about a different cluster, or type 'exit' to quit: "
    else:
        prompt = """
        Hi, I'm Easy Rawlins!
        I'm a bot you can ask questions about SRC's Sherlock and Farmshare docs.
        I'm not perfect, though, so if what I tell you doesn't work, or if I can't find anything for you, don't hesitate to reach out to srcc-support@stanford.edu.
        Which cluster are you using and what would you like to know about it? (or type 'exit' to quit): """
    
    user_query = input(prompt).strip()
    
    if user_query.lower() == 'exit':
        print("Exiting the interactive query session.")
        break

    potential_new_cluster = identify_cluster(user_query)
    
    if potential_new_cluster != "unknown" and potential_new_cluster != current_cluster:
        current_cluster = potential_new_cluster

    print(f"DEBUG: Identified cluster - {current_cluster}")

    if current_cluster in agents:
        retriever_tool = retriever_tools[current_cluster]
        retrieved_docs = retriever_tool.forward(user_query)

        if not retrieved_docs.strip():
            print(f"No relevant information found in {current_cluster.capitalize()} docs. Please contact srcc-support@stanford.edu for more help!")
            continue

        # Generate and print the response
        from_cache = False
        try:
            agent_output = agents[current_cluster].model(user_query, retrieved_docs)
            from_cache = not agent_output is None  # Detect if data returned from cache
        except Exception as e:
            print(f"Error in generating model output: {e}")
            continue

        if agent_output:
            if "### Response:" in agent_output:
                response = agent_output.split("### Response:", 1)[1].strip()
                if response:
                    print(response)
                else:
                    if not from_cache:
                        print(f"No relevant information found in {current_cluster.capitalize()} docs. Please contact srcc-support@stanford.edu for more help!")
            else:
                if not from_cache:
                    print(f"No relevant information found in {current_cluster.capitalize()} docs. Please contact srcc-support@stanford.edu for more help!")
        else:
            if not from_cache:
                print(f"No relevant information found in {current_cluster.capitalize()} response. Please contact srcc-support@stanford.edu for more help!")

    else:
        print("Could not identify the relevant system for the query. Please specify the cluster or check your wording.")
        current_cluster = None  # Reset the cluster if identification fails
