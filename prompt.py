def build_prompt(query, chunks):
    prompt = """You are QuantGPT, a quantitative finance expert.
Answer the following clearly and directly using ONLY the context. Think twice before you answer and do not make any assumptions.
Context:\n"""
    for i, chunk in enumerate(chunks[:25]):
        prompt=prompt+f"\n[Chunk {i+1}]\n{chunk}\n"
    prompt=prompt+f"\n---\nAnswer the following question based only on the context above:\n\n{query}\n"
    return prompt


