import json
from llm import get_llm
from prompts import MEMORY_PROMPT

MAX_TURNS = 5
MAX_CODES = 3


class Memory:
    def __init__(self):
        self.turns = []          
        self.codes = []          
        self.preferences = {"language": "", "framework": "", "topics": []}

    def add_turn(self, question, answer):
        self.turns.append((question, answer))
        self.turns = self.turns[-MAX_TURNS:]

    def add_code(self, code):
        self.codes.append(code)
        self.codes = self.codes[-MAX_CODES:]

    def context(self):
        lines = []
        if self.preferences["language"]:
            lines.append(f"Preferred language: {self.preferences['language']}")
        if self.preferences["framework"]:
            lines.append(f"Preferred framework: {self.preferences['framework']}")
        if self.preferences["topics"]:
            lines.append(f"Topics discussed: {', '.join(self.preferences['topics'])}")
        for question, _ in self.turns[-2:]:
            lines.append(f"Earlier question: {question}")
        return "\n".join(lines) if lines else "No previous context."

    def update(self, query, language, intent):
        # small LLM call that merges the new exchange into the preferences JSON
        llm = get_llm(temperature=0)
        prompt = MEMORY_PROMPT.format(
            preferences=json.dumps(self.preferences),
            query=query,
            language=language,
            intent=intent,
        )
        reply = llm.invoke(prompt).content.strip()
        start = reply.find("{")
        end = reply.rfind("}") + 1
        if start != -1:
            self.preferences = json.loads(reply[start:end])
