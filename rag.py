import argparse
from transformers import AutoTokenizer
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from tqdm import tqdm
from transformers.agents import ReactJsonAgent
import logging

from utils.ollama_engine import OllamaEngine
# from IPython.display import display, Markdown
from utils.openai_engine import OpenAIEngine
from utils.retriever_tool import RetrieverTool

logger = logging.getLogger(__name__)

##################
# FUNCTIONS
##################
def load_content_files(content_dir: str) -> list[Document]:
    from pathlib import Path
    import re
    METADATA_IGNORE = ['', 'Users', 'johnday', 'repos', 'md', 'Documents', 'projects', 'project-bob', 'data']

    def clean_text(file) -> str:
        with open(file, 'r') as f:
            txt = f.read()

        txt = re.sub(r'!?\[.+]\(.+\)\n?', '', txt)  # remove images
        txt = re.sub(r'\n\|.*', '', txt)  # remove tables
        return txt

    def get_data(md_dir) -> list[Document]:
        # Document {page_content, metadata}
        def metadata_by_path(path: str) -> list[str]:
            node = [x for x in path.split("/") if x not in METADATA_IGNORE and not x.endswith('.md')]
            return node

        docs = []
        for file in list(Path(md_dir).rglob('*.md')):
            doc = Document(page_content=clean_text(file), metadata=dict(id=str(file), title=file.stem, tags=metadata_by_path(str(file))))
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
def pipeline(args):
    confirm = input("Are you sure you want to reindex the database? (y/[n]): ")
    if not confirm or confirm.lower() != "y":
        print("Exiting pipeline...")
        return
    source_docs = load_content_files(args.content_path_dir)
    print()
    logger.info(f"source_docs[0]:  {source_docs[0]}")

    tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", "  ", " ", ""],
    )

    # Split documents and remove duplicates
    print(f"Splitting {len(source_docs)} documents...")
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
    vectordb.save_local(args.index_dir)
    print("Vector database created successfully")

def load_vectordb(vectordb_path: str, embeddings: Embeddings) -> FAISS:
    return FAISS.load_local(folder_path=vectordb_path, embeddings=embeddings, allow_dangerous_deserialization=True)

def print_metadata(metadata) -> None:
    print("METADATA:")
    logger.debug(f"metadata: {metadata}")
    metadata_ids = []
    i = 1
    for index, entry in enumerate(metadata, start=1):
        if entry['id'] not in metadata_ids:
            print(f"{i}. {entry['title']}:  \"{entry['id']}\"")
            print(f"     tags: {entry['tags']}")
        metadata_ids.append(entry['id'])
        i += 1

def init(args) -> None:
    # https://docs.python.org/3/howto/logging.html
    # assuming loglevel is bound to the string value obtained from the
    # command line argument. Convert to upper case to allow the user to
    # specify --log=DEBUG or --log=debug
    numeric_level = getattr(logging, args.log.upper(), 'ERROR')
    logger.debug(f"numeric_level: {numeric_level}")
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.log)
    logging.basicConfig(level=numeric_level)

    # remove cache

def llm_engine_factory(model_name: str, temperature: float):
    if model_name.startswith('gpt'):
        return OpenAIEngine(temperature=temperature, model_name=model_name)
    if model_name == "llama3" or model_name == "ollama":
        return OllamaEngine(temperature=temperature, model_name=model_name)
    raise ValueError(f"Unknown model name: {model_name}")

# MAIN
def main(args) -> None:
    if args.db_refresh:
        pipeline(args)
        exit(0)
    else:
        vectordb = load_vectordb(args.index_dir, embedding_model())

    retriever_tool = RetrieverTool(vectordb, k=args.retriver_k)
    # llm_engine = OpenAIEngine(temperature=args.temperature, model_name=args.model)
    llm_engine = llm_engine_factory(args.model, args.temperature)

    while True:
        question = input("QUESTION: ")
        question_lower = question.lower()
        if question_lower == "exit" or question_lower == "quit" or question_lower == "q" or question_lower == "":
            break

        agent = ReactJsonAgent(tools=[retriever_tool], llm_engine=llm_engine, max_iterations=args.max_iterations)
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
    BASE_INDEX_DIR = "/Users/johnday/repos/md/indexes"
    BASE_CONTENT_PATH_DIR = "/Users/johnday/repos/md/data"

    HYBRIS_CONTENT_PATH_DIR = BASE_CONTENT_PATH_DIR
    HYBRIS_INDEX_DIR = f"{BASE_INDEX_DIR}/faiss_index_hybris"

    MEETING_CONTENT_PATH_DIR = "/Users/johnday/Documents/projects/project-bob/transcripts/meetings"
    MEETING_INDEX_DIR = f"{BASE_INDEX_DIR}/faiss_index_meeting"

    # https://docs.python.org/3/library/argparse.html#default
    parser = argparse.ArgumentParser(description="RAG using agentic huggingface")
    parser.add_argument("--db_refresh", action="store_true", help="Reindex vector database?")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature")
    parser.add_argument("--retriver_k", type=int, default=7, help="Retriever tool k")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name")
    parser.add_argument("--log", type=str, default="ERROR", help="Log level")
    parser.add_argument("--db", type=str, choices=['meeting', 'hybris'], default="hybris", help="Database to use")
    parser.add_argument("--content_path_dir", type=str, default=HYBRIS_CONTENT_PATH_DIR, help="Path to content directory")
    parser.add_argument("--index_dir", type=str, default=HYBRIS_INDEX_DIR, help="Path to index")
    parser.add_argument("--max_iterations", type=str, default=6, help="Max iterations")
    parser.add_argument("--chunk_size", type=int, default=200, help="Chunk size")
    parser.add_argument("--chunk_overlap", type=int, default=40, help="Chunk overlap")
    args = parser.parse_args()
    if args.db == "meeting":
        args.content_path_dir = MEETING_CONTENT_PATH_DIR
        args.index_dir = MEETING_INDEX_DIR
    print(args)
    print()

    init(args)

    main(args)
