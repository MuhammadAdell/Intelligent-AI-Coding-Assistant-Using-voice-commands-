# Run with: python3 -m uvicorn server:app --port 8000
# Endpoints and request samples: http://localhost:8000/docs

import tempfile
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel

from assistant import code_language, extract_code, respond
from code_runner import run_code
from knowledge import add_solution, get_collection
from memory import Memory
from stt import transcribe

app = FastAPI(title="AI Coding Assistant")

SESSIONS = {}


class Message(BaseModel):
    message: str


class IndexRequest(BaseModel):
    index: int


class FeedbackRequest(BaseModel):
    index: int
    solution: str


def model_error_text(error):
    text = str(error)
    if "429" in text or "Rate limit" in text:
        return (
            "OpenRouter free-tier request limit reached (50 requests/day). "
            "It resets at midnight UTC — please try again after that."
        )
    return f"The model call failed: {text[:300]}"


def session_of(request: Request, response: Response):
    session_id = request.cookies.get("assistant_session")
    if not session_id or session_id not in SESSIONS:
        session_id = uuid.uuid4().hex
        SESSIONS[session_id] = {"memory": Memory(), "messages": []}
        response.set_cookie(
            "assistant_session", session_id, httponly=True, samesite="strict"
        )
    return SESSIONS[session_id]


def state(session):
    memory = session["memory"]
    return {
        "messages": session["messages"],
        "preferences": memory.preferences,
        "turns": [question for question, _ in memory.turns],
        "codes": len(memory.codes),
        "chunks": get_collection().count(),
    }


def message_at(session, index):
    try:
        return session["messages"][index]
    except IndexError:
        raise HTTPException(status_code=404, detail="no such message")


def answer_in_session(session, query):
    """Run one question through the assistant and return the new chat state."""
    memory = session["memory"]
    session["messages"].append({"role": "user", "content": query})
    try:
        result = respond(query, memory)
        answer = "".join(part for part in result["stream"])
    except Exception as error:
        # keep the failure in the history, an error is not a memory entry
        session["messages"].append(
            {"role": "assistant", "content": model_error_text(error)}
        )
        return state(session)

    message = {
        "role": "assistant",
        "content": answer,
        "intent": result["intent"],
        "code": extract_code(answer),
        "language": code_language(answer),
        "sources": result["sources"],
        "needs_feedback": not result["relevant"],
    }
    session["messages"].append(message)
    try:
        memory.add_turn(query, answer)
        if message["code"]:
            memory.add_code(message["code"])
        memory.update(query, message["language"], result["intent"])
    except Exception as error:
        message["memory_error"] = model_error_text(error)
    return state(session)


def audio_temp_file(data, filename):
    suffix = Path(filename or "").suffix or ".wav"
    audio = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    audio.write(data)
    audio.close()
    return Path(audio.name)


@app.get("/assistant")
def assistant_state(session=Depends(session_of)):
    return state(session)


@app.post("/chat")
def chat(body: Message, session=Depends(session_of)):
    query = body.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="empty message")
    return answer_in_session(session, query)


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    audio = audio_temp_file(await file.read(), file.filename)
    try:
        return {"text": transcribe(str(audio))}
    finally:
        audio.unlink(missing_ok=True)


@app.post("/voice")
async def voice_chat(file: UploadFile = File(...), session=Depends(session_of)):
    audio = audio_temp_file(await file.read(), file.filename)
    try:
        question = transcribe(str(audio))
    finally:
        audio.unlink(missing_ok=True)
    if not question:
        raise HTTPException(status_code=400, detail="no speech detected")
    return answer_in_session(session, question)


@app.post("/run-code")
def execute_code(body: IndexRequest, session=Depends(session_of)):
    message = message_at(session, body.index)
    if not message.get("code"):
        raise HTTPException(status_code=400, detail="this reply has no code")
    message["output"], message["stderr"] = run_code(message["code"])
    return state(session)


@app.post("/feedback")
def teach(body: FeedbackRequest, session=Depends(session_of)):
    message = message_at(session, body.index)
    message["learned"] = add_solution(body.solution.strip())
    message["needs_feedback"] = False
    return state(session)


@app.post("/reset")
def reset(session=Depends(session_of)):
    session["memory"] = Memory()
    session["messages"] = []
    return state(session)
