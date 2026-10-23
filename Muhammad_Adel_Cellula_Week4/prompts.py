# All prompt templates in one place so they are easy to tune.

CLASSIFIER_PROMPT = """You are an intent classifier for an AI coding assistant.
Read the user request and decide its class.

Classes:
- EXPLAIN: the user pasted code or points at code and wants it analyzed,
  described line by line, or debugged. No new program is requested.
- GENERATE: the user wants new code written, e.g. "write", "create",
  "build", "implement", "generate a CNN model", "sorting algorithm".

Rule: EXPLAIN requires an actual code snippet inside the request.
If the request contains no code, answer GENERATE.
Words like "explain", "analyze", "what does this do", "debug" together with
a pasted snippet mean EXPLAIN.

Answer with exactly one word: EXPLAIN or GENERATE

User request:
{query}
"""

EVALUATOR_PROMPT = """You are a retrieval relevance evaluator in a RAG pipeline.
The user asked a programming question. We retrieved these documents from the
knowledge base:

{context}

Question: Is the retrieved context relevant enough to answer the user's
request correctly? Judge only relevance, not writing quality.

Answer with exactly one word: Relevant or Not Relevant

User request:
{query}
"""

NO_KNOWLEDGE_REPLY = (
    "I couldn't find relevant knowledge for your request. "
    "Could you provide the correct solution or reference so I can learn it "
    "for future interactions?"
)

EXPLAIN_PROMPT = """You are a code explanation assistant.
Analyze the code and explain what it does, how data flows through it, and
point out anything incorrect or risky. Use short paragraphs and markdown.
Do not rewrite the whole program unless asked.

Conversation memory:
{memory}

User request:
{query}
"""

GENERATE_PROMPT = """You are a code generation assistant working with RAG.
Use the retrieved documents below as reference material. Produce:
- Complete runnable source code in one markdown python block
- Comments on the important steps
- Error handling where input can fail
- Best practices from the retrieved documents
After the code add a short explanation (max 5 lines).

Retrieved documents:
{context}

Conversation memory:
{memory}

User request:
{query}
"""

MEMORY_PROMPT = """You maintain the long term memory of a coding assistant.
Current stored preferences (JSON):
{preferences}

Latest exchange:
User: {query}
Assistant used: {language} and topic {intent}

Update the preferences JSON with anything new about the user: programming
language preference, framework preference, project topics, style choices.
Keep keys: language, framework, topics. Reply with only the JSON object.
"""
