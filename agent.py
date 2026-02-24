"""
This is an Enterprise Knowledge agent.

"""

from core import load_components, retrieve, build_prompt, generate_answer

def format_response(answer: str, chunks: list) -> str:
    output = []
    output.append(f"\n{'='*60}")
    output.append("  ANSWER")
    output.append(f"{'='*60}")
    output.append(answer)
    output.append(f"\n{'─'*60}")
    output.append("  RETRIEVED SOURCES")
    output.append(f"{'─'*60}")

    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        similarity = round(1 - chunk["distance"], 3)
        output.append(
            f"  [{i+1}] {meta['filename']} | "
            f"Page {meta['page_num']} | "
            f"Lines {meta['line_start']}–{meta['line_end']} | "
            f"Similarity: {similarity}"
        )

    output.append(f"{'='*60}\n")
    return "\n".join(output)

# ---- Main Agent Loop ----

def run_agent():
    print(f"\n{'='*60}")
    print("  MAHESH'S SOLUTIONS KNOWLEDGE AGENT")
    print("  Ask questions about company policies and documents.")
    print("  Type 'exit','q','quit','end' to quit.")
    print(f"{'='*60}\n")

    """loading everything once i.e following tuple unpacking as
    load_components returns 3 (embedder, collection, groq_client)"""

    embedder, collection, groq_client = load_components()

    while True:
        question = input("Your Question:  ").strip()

        if not question:
            continue

        if question.lower() in ['exit','quit','q','Quit','end','End']:
            print("Good Bye dude, I was Happy clarifying you")
            break

        print(f"\n Searching Documents ....\n ")
        
        # Pipeline: retrieve -> build prompt -> generate -> format
        chunks = retrieve(question,embedder,collection)
        prompt = build_prompt(question, chunks)
        answer = generate_answer(prompt, groq_client)
        output = format_response(answer, chunks)

        print(output)


if __name__ == "__main__":
    run_agent()        

    

