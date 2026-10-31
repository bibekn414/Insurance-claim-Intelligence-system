from backend.inference import predict_image, predict_claim
from backend.fusion_engine import fuse_predictions
from backend.rag_engine import (
    build_query,
    retrieve_policy_clauses,
    format_retrieved_clauses
)
from backend.gemini_client import generate_report


def run_pipeline(image_path, claim_text):
    """
    Complete Multimodal Insurance Claim Pipeline
    """

    # Step 1: Image Prediction
    image_result = predict_image(image_path)

    # Step 2: Claim Text Prediction
    text_result = predict_claim(claim_text)

    # Step 3: Fusion
    fusion_result = fuse_predictions(
        image_result,
        text_result
    )

    # Step 4: Build RAG Query
    query = build_query(fusion_result)

    # Step 5: Retrieve Policy Clauses
    retrieved_docs = retrieve_policy_clauses(query)

    rag_output = format_retrieved_clauses(retrieved_docs)

    policy_context = rag_output["context"]

    sources = rag_output["sources"]

    # Step 7: Generate report
    report = generate_report(
        fusion_result,
        policy_context
    )


    return {
        "fusion_result": fusion_result,
        "retrieved_documents": retrieved_docs,
        "policy_context": policy_context,
        "sources": sources,
        "report": report
    }


if __name__ == "__main__":

    image_path = "sample_images/car2.jpg"

    claim_text = """
    While driving, a stone hit the windshield causing the
    front glass to crack completely.
    """

    result = run_pipeline(image_path, claim_text)

    import json

    print("FUSION RESULT")  
    print(json.dumps(result["fusion_result"], indent=4))

    print("\n")

    print("RAG SOURCES")
    print(json.dumps(result["sources"], indent=4))

    print("\n")
    print("RETRIEVED POLICY CLAUSES")
    print(result["policy_context"])

    print("\n")

    print("FINAL REPORT")
    print(json.dumps(result["report"], indent=4))