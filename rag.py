from transformers import AutoTokenizer
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from tqdm import tqdm
from transformers.agents import Tool, HfEngine, ReactJsonAgent
from huggingface_hub import InferenceClient
import logging
from IPython.display import display, Markdown

from utils.openai_engine import OpenAIEngine
from utils.retriever_tool import RetrieverTool

# Set up logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# PIPELINE
CONTENT_PATH_DIR = "/Users/johnday/repos/md"
INDEX_DIR = f"{CONTENT_PATH_DIR}/faiss_index_hybris"
MAX_ITERATIONS = 6
VERBOSE = 0
chunk_size = 200
chunk_overlap = 20

# RUNTIME
question = "How can I task server"
# retriever k at retriever_tool
TEMPERATURE = 0.5

##################
# FUNCTIONS
##################
def load_content_files(content_dir: str) -> list[Document]:
    from pathlib import Path
    import re

    def clean_text(file) -> str:
        with open(file, 'r') as f:
            txt = f.read()

        txt = re.sub(r'!?\[.+]\(.+\)\n?', '', txt)  # remove images
        txt = re.sub(r'\n\|.*', '', txt)  # remove tables
        return txt

    def get_data(md_dir) -> list[Document]:
        # Document {page_content, metadata}
        def metadata_by_path(path: str) -> list[str]:
            node = [x for x in path.split("/") if x != '' and x != 'Users' and x != 'johnday' and x != 'repos' and x != 'md' and not x.endswith('.md')]
            # return comma delimited string from node
            return node

        docs = []
        for file in list(Path(md_dir).rglob('*.md')):
            doc = Document(page_content=clean_text(file), metadata=dict(id=str(file), title=file.stem, subject=metadata_by_path(str(file))))
            docs.append(doc)
        return docs

    return get_data(content_dir)

def run_agentic_rag(question: str, agent) -> str:
    # Function to run the agent
    enhanced_question = f"""Using the information contained in your knowledge base, which you can access with the 'retriever' tool,
give a comprehensive answer to the question below.
Respond only to the question asked, response should be concise and relevant to the question.
If you cannot find information, do not give up and try calling your retriever again with different arguments!
Make sure to have covered the question completely by calling the retriever tool several times with semantically different queries.
Your queries should not be questions but affirmative form sentences: e.g. rather than "How do I load a model from the Hub in bf16?", query should be "load a model from the Hub bf16 weights".

Question:
{question}"""

    return agent.run(enhanced_question)

def embedding_model():
    return HuggingFaceEmbeddings(model_name="thenlper/gte-small")
def pipeline():
    source_docs = load_content_files(CONTENT_PATH_DIR)
    print(len(source_docs))
    print()
    # print(source_docs[0])

    tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", "  ", " ", ""],
    )

    # Split documents and remove duplicates
    print("Splitting documents...")
    docs_processed = []
    unique_texts = {}
    for doc in tqdm(source_docs):
        new_docs = text_splitter.split_documents([doc])
        for new_doc in new_docs:
            if new_doc.page_content not in unique_texts:
                unique_texts[new_doc.page_content] = True
                docs_processed.append(new_doc)
    print(f"Processed {len(docs_processed)} unique document chunks")

    # Create the vector database
    # https://python.langchain.com/v0.2/docs/integrations/vectorstores/faiss/
    print("Creating vector database...")
    vectordb = FAISS.from_documents(
        documents=docs_processed,
        embedding=embedding_model(),
        distance_strategy=DistanceStrategy.COSINE,
    )
    vectordb.save_local(INDEX_DIR)
    print("Vector database created successfully")

    return vectordb

def load_vectordb(vectordb_path: str, embeddings: Embeddings) -> FAISS:
    return FAISS.load_local(folder_path=vectordb_path, embeddings=embeddings, allow_dangerous_deserialization=True)


# MAIN
def main(args) -> None:
    if args.db_refresh:
        vectordb = pipeline()
    else:
        vectordb = load_vectordb(f"{INDEX_DIR}", embedding_model())

    retriever_tool = RetrieverTool(vectordb)

    llm_engine = OpenAIEngine(temperature=TEMPERATURE)

    agent = ReactJsonAgent(tools=[retriever_tool], llm_engine=llm_engine, max_iterations=MAX_ITERATIONS, verbose=VERBOSE)

    while True:
        question = input("Ask a question: ")
        if question.lower() == "exit" or question.lower() == "quit" or question.lower() == "q":
            break

        answer = run_agentic_rag(question, agent)
        print(f"Question: {question}")
        print(f"Answer: {answer}\n\n")

##################
# ENTRY POINT
##################
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG using agentic huggingface")
    parser.add_argument("--db_refresh", action="store_true", help="Reindex vector database?")
    # parser.add_argument("--cleanup", action="store_true", help="Cleanup files on success?")
    # parser.add_argument("--s3_bucket", type=str, default=S3_BUCKET_IN, help="S3 bucket")
    args = parser.parse_args()

    main(args)
