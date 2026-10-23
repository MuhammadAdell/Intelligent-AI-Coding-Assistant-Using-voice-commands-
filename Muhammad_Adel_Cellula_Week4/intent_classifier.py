from llm import get_llm
from prompts import CLASSIFIER_PROMPT

def classify_intent(query):
    llm = get_llm(temperature=0)
    reply = llm.invoke(CLASSIFIER_PROMPT.format(query=query)).content.strip().upper()
    if "GENERATE" in reply:
        return "generate"
    if "EXPLAIN" in reply:
        return "explain"
    return "generate"
