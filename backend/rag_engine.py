import os
import logging

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY_FOLDER = os.path.join(BASE_DIR, "insurance_policies")


VECTOR_DB = os.path.join(BASE_DIR, "policy_db")

##Creating the embedding model
def get_embedding_model():
    logger.info("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings

def build_vector_database(chunks):
    logger.info("Creating vector database...")
    embeddings = get_embedding_model()
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB
    )
    logger.info("Vector database created successfully.")
    return vectordb


def load_documents():
    logger.info("Loading insurance policy PDFs")
    loader = PyPDFDirectoryLoader(POLICY_FOLDER)
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} pages")
    return documents


def load_vector_database():

    logger.info("Loading existing vector database")
    embeddings = get_embedding_model()
    vectordb = Chroma(
        persist_directory=VECTOR_DB,
        embedding_function=embeddings
    )
    return vectordb


def retrieve_policy_clauses(query, k=5):
    vectordb = load_vector_database()
    results = vectordb.max_marginal_relevance_search(
    query=query,
    k=3,
    fetch_k=20,
    lambda_mult=0.7
)
    return results


def split_documents(documents):
    logger.info("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks.")
    return chunks

chunk_size = 800
chunk_overlap = 150


def build_query(fusion_result):
    """
    Build an intelligent search query from the Fusion Engine output.
    """

    # Get top predicted damages
    top_damages = list(fusion_result["final_damage"].keys())[:3]

    query_parts = [
        "private car insurance policy",
        "coverage",
        "claim settlement",
        "accidental damage",
        "exclusions"
    ]

    query_parts.extend(top_damages)

    query_parts.append(
        fusion_result["recommended_action"]
    )

    query_parts.append(
        fusion_result["conflict_level"]
    )

    return " ".join(query_parts)


def format_retrieved_clauses(results):
    """
    Convert retrieved documents into readable text for Gemini.
    """
    formatted = []

    for i, doc in enumerate(results, start=1):
        source = os.path.basename(doc.metadata["source"])
        page = doc.metadata["page"] + 1
        formatted.append(
            f"""
Policy {i}
Source : {source}
Page : {page}
Clause:
{doc.page_content}
"""
        )

    return "\n\n".join(formatted)

if __name__ == "__main__":

    query = build_query(fusion_result)
    results = retrieve_policy_clauses(query)

    print("Retrieved Clauses")


    for i, doc in enumerate(results, start=1):
        print(f"\nResult {i}")
        print("Source :", doc.metadata["source"])
        print("Page   :", doc.metadata["page"] + 1)
        print()
        print(doc.page_content[:600])

def retrieve_policy_clauses(query, k=5):

    vectordb = load_vector_database()

    results = vectordb.max_marginal_relevance_search(
        query=query,
        k=k,
        fetch_k=20,
        lambda_mult=0.7
    )

    return results


def format_retrieved_clauses(results):

    formatted = []

    sources = []

    for doc in results:

        source = os.path.basename(
            doc.metadata["source"]
        )

        page = doc.metadata["page"] + 1

        sources.append({
            "source": source,
            "page": page
        })

        formatted.append(
            f"""
Source: {source}
Page: {page}

{doc.page_content}
"""
        )

    return {
        "context": "\n\n".join(formatted),
        "sources": sources
    }