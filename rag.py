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
from transformers.agents import ReactJsonAgent
from huggingface_hub import InferenceClient
import logging
# from IPython.display import display, Markdown

from utils.openai_engine import OpenAIEngine
from utils.retriever_tool import RetrieverTool

# Set up logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# PIPELINE
CONTENT_PATH_DIR = "/Users/johnday/repos/md"
INDEX_DIR = f"{CONTENT_PATH_DIR}/faiss_index_hybris"
MAX_ITERATIONS = 6
VERBOSE = -1
chunk_size = 200
chunk_overlap = 20

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
    def clean_answer(answer: str) -> str:
        return answer.replace("**", "")

    # Function to run the agent
    enhanced_question = f"""Using the information contained in your knowledge base, which you can access with the 'retriever' tool,
give a comprehensive answer to the question below.
Respond only to the question asked, response should be concise and relevant to the question.
If you cannot find information, do not give up and try calling your retriever again with different arguments!
Make sure to have covered the question completely by calling the retriever tool several times with semantically different queries.
Your queries should not be questions but affirmative form sentences: e.g. rather than "How do I load a model from the Hub in bf16?", query should be "load a model from the Hub bf16 weights".

Question:
{question}"""

    return clean_answer(agent.run(enhanced_question))

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

def print_metadata(metadata):
    print("METADATA:")
    metadata_ids = []
    i = 1
    for index, entry in enumerate(metadata, start=1):
        if entry['id'] not in metadata_ids:
            print(f"{i}. {entry['title'].replace('-', ' ').title()}:  \"{entry['id']}\"\n")
        metadata_ids.append(entry['id'])
        i += 1


# MAIN
def main(args) -> None:
    if args.db_refresh:
        vectordb = pipeline()
    else:
        vectordb = load_vectordb(f"{INDEX_DIR}", embedding_model())

    retriever_tool = RetrieverTool(vectordb, k=args.retriver_k)
    llm_engine = OpenAIEngine(temperature=args.temperature, model_name=args.model)

    while True:
        question = input("QUESTION: ")
        question_lower = question.lower()
        if question_lower == "exit" or question_lower == "quit" or question_lower == "q" or question_lower == "":
            break

        agent = ReactJsonAgent(tools=[retriever_tool], llm_engine=llm_engine, max_iterations=MAX_ITERATIONS, verbose=VERBOSE)
        agent.logger.setLevel(logging.CRITICAL)

        answer = run_agentic_rag(question, agent)
        print(f"\nANSWER: {answer}\n")
        print_metadata(retriever_tool.metadata)
        retriever_tool.reset_metadata()
        print()
        print()

##################
# ENTRY POINT
##################
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG using agentic huggingface")
    parser.add_argument("--db_refresh", action="store_true", help="Reindex vector database?")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature")
    parser.add_argument("--retriver_k", type=int, default=7, help="Retriever tool k")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name")
    # parser.add_argument("--cleanup", action="store_true", help="Cleanup files on success?")
    parser.add_argument("--log", type=str, default="ERROR", help="Log level, e.g. DEBUG, INFO, WARNING, ERROR, CRITICAL")
    args = parser.parse_args()
    print(args)
    print()

    # https://docs.python.org/3/howto/logging.html
    # assuming loglevel is bound to the string value obtained from the
    # command line argument. Convert to upper case to allow the user to
    # specify --log=DEBUG or --log=debug
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.log)
    logging.basicConfig(level=numeric_level)

    main(args)
