"""
eval.py - Experiment tracking for this rag (mlflow)
Runs the test questions and tracks quality metrics

"""

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

import mlflow.artifacts
from core import load_components, retrieve, build_prompt, generate_answer

TEST_QUESTIONS = [
    {
        "question": "How many sick days do employees get per year ?",
        "expected_keywords": ["sick","days","12","leave"]
    },
    {
        "question": "What is the hotel expense limit for travel ?",
        "expected_keywords": ["hotel","200","night","expense"]
    },
    {
        "question": "What are the escalation levels for client complaints ?",
        "expected_keywords": ["escalation","level","client"]
    },
    {
        "question": "what are decided in the Q3 product meeting ?",
        "expected_keywords": ["Q3","meeting","product","decision"]
    },
    {
        "question": "What are the remote work anchor day ?",
        "expected_keywords": ["tuesday","thursday","anchor","remote"]
    },
]

# METRICS

def avg_similarity(chunks: list) -> float:
    if not chunks:
        return 0.0
    scores = [round(1-chunk["distance"],3) for chunk in chunks]
    return round(sum(scores)/len(scores),3)

def top1_similarity(chunks: list) -> float:
    if not chunks:
        return 0.0
    return round(1-chunks[0]["distance"],3)

def keyword_hit_rate(answer: str, keywords: list) -> float:
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return round(hits/ len(keywords), 3)

# Experiment Runner

def run_experiment(
        chunk_size: int = 20,
        overlap: int = 4,
        top_k: int = 5,
        experiment_name: str = "RAG PipeLine Evaluation"
):
    """ Run Evaluation and log results to MLflow"""
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"chunk{chunk_size}_overlap{overlap}_topk{top_k}"):
        
        mlflow.log_param("chunk_size", chunk_size)
        mlflow.log_param("overlap",overlap)
        mlflow.log_param("top_k",top_k)
        mlflow.log_param("embed_model","all-MiniLM-L6-v2")
        mlflow.log_param("llm_model", "llama-3.3-70b-versatile")
        mlflow.log_param("num_test_questions", len(TEST_QUESTIONS))

        print("\n Loading Components ___")
        embedder, collection, groq_client = load_components()

        #Run each test question
        all_avg_sim = []
        all_top1_sim = []
        all_keyword_hits = []

        for i, test in enumerate(TEST_QUESTIONS):
            question = test["question"]
            keywords = test["expected_keywords"]

            print(f"\n[{i+1}/{len(TEST_QUESTIONS)}] {question}")

            # Retrieve
            chunks = retrieve(question, embedder, collection)

            # Generate answer
            prompt = build_prompt(question, chunks)
            answer = generate_answer(prompt, groq_client)

            # Calculate metrics
            avg_sim    = avg_similarity(chunks)
            top1_sim   = top1_similarity(chunks)
            kw_hit     = keyword_hit_rate(answer, keywords)

            all_avg_sim.append(avg_sim)
            all_top1_sim.append(top1_sim)
            all_keyword_hits.append(kw_hit)

            print(f"  Avg similarity : {avg_sim}")
            print(f"  Top1 similarity: {top1_sim}")
            print(f"  Keyword hits   : {kw_hit}")

            # Log per-question metrics
            mlflow.log_metric(f"q{i+1}_avg_similarity", avg_sim)
            mlflow.log_metric(f"q{i+1}_top1_similarity", top1_sim)
            mlflow.log_metric(f"q{i+1}_keyword_hit_rate", kw_hit)

        # Log overall metrics
        overall_avg_sim    = round(sum(all_avg_sim) / len(all_avg_sim), 3)
        overall_top1_sim   = round(sum(all_top1_sim) / len(all_top1_sim), 3)
        overall_kw_hits    = round(sum(all_keyword_hits) / len(all_keyword_hits), 3)

        mlflow.log_metric("overall_avg_similarity", overall_avg_sim)
        mlflow.log_metric("overall_top1_similarity", overall_top1_sim)
        mlflow.log_metric("overall_keyword_hit_rate", overall_kw_hits)

        print(f"\n{'='*55}")
        print(f"  EXPERIMENT COMPLETE")
        print(f"{'='*55}")
        print(f"  chunk_size             : {chunk_size}")
        print(f"  overlap                : {overlap}")
        print(f"  top_k                  : {top_k}")
        print(f"  Overall avg similarity : {overall_avg_sim}")
        print(f"  Overall top1 similarity: {overall_top1_sim}")
        print(f"  Overall keyword hits   : {overall_kw_hits}")
        print(f"{'='*55}\n")


if __name__ == "__main__":

    print("Running RAG Pipeline Experiments...")
    print("Results will be tracked in MLflow\n")

    # Experiment 1 — baseline
    run_experiment(chunk_size=20, overlap=4, top_k=5)

    # Experiment 2 — larger chunks
    run_experiment(chunk_size=30, overlap=6, top_k=5)

    # Experiment 3 — more retrieved chunks
    run_experiment(chunk_size=20, overlap=4, top_k=8)

    print("\nAll experiments done!")
    print("Run: mlflow ui")
    print("Then open: http://localhost:5000")
