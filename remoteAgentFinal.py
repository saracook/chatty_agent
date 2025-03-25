import os
from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
from smolagents import Tool, HfApiModel, CodeAgent

load_dotenv()

# Define paths to the directories containing the markdown files for different clusters
clusters = {
    "sherlock": "/scratch/users/bcritt/sherlockMDs/",
    "farmshare": "/scratch/users/bcritt/farmshareDocs/"
}

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
                "The query to perform. This should be semantically close to your "
                "target documents. Use the affirmative form rather than a question."
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
        return "\nRetrieved documents:\n" + "".join(
            [f"\n\n===== Document {i} =====\n" + doc.page_content for i, doc in enumerate(docs)]
        )

# Initialize agents for each cluster dynamically
agents = {}
for cluster_name, path in clusters.items():
    docs = ingest_markdown_files(path)
    docs_processed = RecursiveCharacterTextSplitter(chunk_size=500).split_documents(docs)[:1000]
    retriever_tool = RetrieverTool(docs_processed)
    agents[cluster_name] = CodeAgent(
        tools=[retriever_tool],
        model=HfApiModel("meta-llama/Llama-3.3-70B-Instruct"),
        max_steps=4
    )

# Function to identify the relevant cluster based on user input
def identify_cluster(user_query: str) -> str:
    """Identify which cluster the user is referring to based on the query."""
    # Use the smolagents library to parse the user's input
    # and identify the relevant cluster
    for cluster_name in clusters:
        if cluster_name in user_query.lower():
            return cluster_name
    return "unknown"

# Main interaction loop
while True:
    user_query = input("Which cluster are you using and what would you like to know about it? (or type 'exit' to quit): ")
    if user_query.lower() == 'exit':
        print("Exiting the interactive query session.")
        break

    # Identify the relevant cluster
    cluster_name = identify_cluster(user_query)

    # Check if the identified cluster is valid
    if cluster_name in agents:
        # Get the agent's response to the user's query from the identified cluster
        agent_output = agents[cluster_name].run(user_query)

        if agent_output.strip():
            print(f"Response from {cluster_name.capitalize()} docs: {agent_output}")
        else:
            print(f"No relevant information found in {cluster_name.capitalize()} docs. Please contact srcc-support@stanford.edu for more help!")
    else:
        print("Could not identify the relevant system for the query. Please specify the cluster or check your wording.")
