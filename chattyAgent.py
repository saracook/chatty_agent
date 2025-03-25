import os
from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
from smolagents import Tool, CodeAgent
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

load_dotenv()

# Define paths to the directories containing the markdown files for different clusters
clusters = {
    "sherlock": "/scratch/users/bcritt/sherlockMDs/",
    "farmshare": "/scratch/users/bcritt/farmshareDocs/"
}

# Update path to the new model
local_model_path="/oak/stanford/groups/ruthm/bcritt/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/e0bc86c23ce5aae1db576c8cca6f06f1f73af2db"

# Function to ingest markdown files from a specified directory
def ingest_markdown_files(corpusdir: str) -> List[Document]:
    documents = []
    for infile in os.listdir(corpusdir):
        if infile.endswith(".md"):  # Assuming your files have .md extension
            with open(os.path.join(corpusdir, infile), 'r', errors='ignore') as fin:
                documents.append(Document(page_content=fin.read(), metadata={"source": infile}))
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
            "description": (
                "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question."
            ),
        }
    }
    output_type = "string"

    def __init__(self, docs: List[Document], **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(docs, k=10)

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"
        docs = self.retriever.invoke(query)
        return "\n".join(
            [f"\n===== Document {i} =====\n{doc.page_content}\n" for i, doc in enumerate(docs)]
        )

# Optimized function to generate text using the model and the retrieved documents
def generate_text(model, tokenizer, user_query, retrieved_docs, max_new_tokens=256, temperature=0.7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Formatting the input with clear instructions
    input_text = f"""
### Task:
Summarize the user's query based on the information provided in the retrieved documents. Include inline citations in the format [Document X] where X represents the document number. Do not provide an answer unless it is supported by the context in the retrieved documents.

### User Query:
{user_query}

### Retrieved Documents:
{retrieved_docs}

### Response:
"""
    # Tokenizing input text
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # Generate the model response
    tokens = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode the tokens to generate the output text
    output_text = tokenizer.decode(tokens[0], skip_special_tokens=True)
    model.to("cpu")  # Move model back to CPU to free GPU memory
    torch.cuda.empty_cache()

    return output_text

# Initialize agents for each cluster dynamically
agents = {}
retriever_tools = {}

for cluster_name, path in clusters.items():
    docs = ingest_markdown_files(path)
    print(f"Ingested {len(docs)} documents for cluster {cluster_name}.")  # Debug message
    docs_processed = RecursiveCharacterTextSplitter(chunk_size=500).split_documents(docs)[:1000]
    retriever_tool = RetrieverTool(docs_processed)
    retriever_tools[cluster_name] = retriever_tool

    try:
        # Load the local model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(local_model_path, torch_dtype=torch.bfloat16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)

        agents[cluster_name] = CodeAgent(
            tools=[retriever_tool],
            model=lambda user_query, retrieved_docs, **kwargs: generate_text(model, tokenizer, user_query, retrieved_docs, **kwargs),
            max_steps=4
        )
        print(f"DEBUG: Loaded agent for cluster - {cluster_name}")  # Debug message
    except Exception as e:
        print(f"Error loading the model or tokenizer from path {local_model_path}: {e}")

# Check populated agents
print(f"DEBUG: Agents loaded - {list(agents.keys())}")  # Debug message

# Function to identify the relevant cluster based on user input
def identify_cluster(user_query: str) -> str:
    user_query_lower = user_query.lower()  # Ensure case insensitivity.

    # Match keywords to identify cluster.
    for cluster_name in clusters:
        if cluster_name in user_query_lower:
            return cluster_name

    return "unknown"

# Main interaction loop
while True:
    user_query = input("""
    Hi, I'm Easy Rawlins!
    I'm a bot of whom you can ask questions about SRC's Sherlock and Farmshare docs.
    I'm not perfect, though, so if what I tell you doesn't work, or if I can't find anything for you, don't hesitate to reach out to srcc-support@stanford.edu.
    Which cluster are you using and what would you like to know about it? (or type 'exit' to quit): """)
    
    # Check for exit command
    if user_query.lower() == 'exit':
        print("Exiting the interactive query session.")
        break

    # Identify the relevant cluster
    cluster_name = identify_cluster(user_query)

    # Debug: Check cluster identification
    print(f"DEBUG: Identified cluster - {cluster_name}")

    # Check if the identified cluster is valid
    if cluster_name in agents:
        # Retrieve documents
        retriever_tool = retriever_tools[cluster_name]
        retrieved_docs = retriever_tool.forward(user_query)

        # Ensure that there is relevant information in the retrieved documents
        if not retrieved_docs.strip():
            print(f"No relevant information found in {cluster_name.capitalize()} docs. Please contact srcc-support@stanford.edu for more help!")
            continue

        # Generate and print the response
        try:
            agent_output = agents[cluster_name].model(user_query, retrieved_docs)
        except Exception as e:
            print(f"Error in generating model output: {e}")
            continue

        # Check if the response is supported by the retrieved documents
        if "### Response:" in agent_output and not agent_output.strip().split("### Response:")[1].strip():
            print(f"No relevant information found in {cluster_name.capitalize()} docs. Please contact srcc-support@stanford.edu for more help!")
        else:
            if agent_output.strip():
                print(agent_output)
            else:
                print(f"No relevant information found in {cluster_name.capitalize()} docs. Please contact srcc-support@stanford.edu for more help!")

    else:
        print("Could not identify the relevant system for the query. Please specify the cluster or check your wording.")
