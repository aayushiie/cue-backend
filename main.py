from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from rag import process_video, ask_question

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoRequest(BaseModel):
    video_id: str

class QuestionRequest(BaseModel):
    question: str

@app.post("/process-video")
def process_video_endpoint(request: VideoRequest):
    return process_video(request.video_id)

@app.post("/ask")
def ask_endpoint(request: QuestionRequest):
    return ask_question(request.question)