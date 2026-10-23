from llm import get_llm
from prompts import EXPLAIN_PROMPT, GENERATE_PROMPT, NO_KNOWLEDGE_REPLY
from intent_classifier import classify_intent
from knowledge import retrieve, evaluate_relevance
from web_search import search_stack_overflow


def extract_code(text):
    parts = text.split("```")
    blocks = parts[1::2]  
    for block in blocks:
        block = block.strip("\n")
        if block.startswith("python"):
            return block.split("\n", 1)[1]
    for block in blocks:
        block = block.strip("\n")
        if "def " in block or "import " in block or "function " in block:
            lines = block.split("\n", 1)
            # drop the language tag left after an untagged python block
            if len(lines) == 2 and " " not in lines[0]:
                return lines[1]
            return block
    return ""


def code_language(text):
    # language tag of the first fenced block, so non-Python answers
    # are recorded as what they actually are
    for tag in (p.split("\n", 1)[0].strip().lower() for p in text.split("```")[1::2]):
        if tag:
            return tag.split()[0]
    return "python" if extract_code(text) else "unknown"


def build_context(documents, sources):
    blocks = []
    for doc, meta in zip(documents, sources):
        blocks.append(f"[from {meta['source']}, chunk {meta['chunk']}]\n{doc}")
    return "\n\n---\n\n".join(blocks)


def respond(query, memory):
    intent = classify_intent(query)

    if intent == "explain":
        llm = get_llm()
        prompt = EXPLAIN_PROMPT.format(memory=memory.context(), query=query)
        return {
            "intent": "explain",
            "used_rag": False,
            "relevant": True,
            "sources": [],
            "stream": (chunk.content for chunk in llm.stream(prompt)),
        }

    documents, sources, _ = retrieve(query)
    context = build_context(documents, sources)

    if not evaluate_relevance(query, context):
        # nothing usable in Chroma, try Stack Overflow before giving up
        documents, sources = search_stack_overflow(query)
        context = build_context(documents, sources)
        if not documents or not evaluate_relevance(query, context):
            def single(text):
                yield text
            return {
                "intent": "generate",
                "used_rag": True,
                "relevant": False,
                "sources": sources,
                "stream": single(NO_KNOWLEDGE_REPLY),
            }

    llm = get_llm()
    prompt = GENERATE_PROMPT.format(
        context=context, memory=memory.context(), query=query
    )
    return {
        "intent": "generate",
        "used_rag": True,
        "relevant": True,
        "sources": sources,
        "stream": (chunk.content for chunk in llm.stream(prompt)),
    }
