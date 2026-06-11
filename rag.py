from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()
from groq import Groq
import os

# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


vector_store = None
transcript_text = ""


def process_video(video_id: str):
    global vector_store
    global transcript_text

    vector_store = None
    transcript_text = ""

    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.fetch(
            video_id,
            languages=["en", "hi"]
        )
    except TranscriptsDisabled:
        return {
            "status": "error",
            "message": "Transcripts are disabled for this video."
        }
    except NoTranscriptFound:
        return {
            "status": "error",
            "message": "No transcript found for this video in English or Hindi."
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to fetch transcript: {str(e)}"
        }
    
    timestamp_map = []  # list of (char_offset, start_seconds)
    parts = []
    offset = 0
    for chunk in transcript_list:
        parts.append(chunk.text)
        timestamp_map.append((offset, chunk.start))
        offset += len(chunk.text) + 1  # +1 for the space join

    transcript_text = " ".join(parts)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )
    raw_chunks = splitter.split_text(transcript_text)

    # Attach the closest timestamp to each chunk via the offset map
    def find_timestamp(chunk_text):
        pos = transcript_text.find(chunk_text[:60])  # anchor on first 60 chars
        if pos == -1:
            return 0
        # Walk timestamp_map to find the last entry whose offset <= pos
        ts = 0
        for char_off, start_sec in timestamp_map:
            if char_off <= pos:
                ts = start_sec
            else:
                break
        return ts

    documents = [
        Document(
            page_content=c,
            metadata={"start": find_timestamp(c)}
        )
        for c in raw_chunks
    ]

    vector_store = FAISS.from_documents(documents, embeddings)

    return {
        "status": "success",
        "chunks": len(documents)
    }


def ask_question(question: str):
    global vector_store
    global transcript_text

    if vector_store is None:
        return {
            "answer": "No video has been processed yet. Please open a YouTube video first.",
            "sources": []
        }

    summary_keywords = ["summary", "summarize", "overview", "key takeaways"]
    is_summary = any(
        keyword in question.lower() for keyword in summary_keywords
    )

    if is_summary:
        context = transcript_text[:12000]
        sources = []
    else:
        docs = vector_store.similarity_search(question, k=4)
        context = "\n\n".join(doc.page_content for doc in docs)
        sources = [
            {
                "timestamp": doc.metadata.get("start", 0),
                "text": doc.page_content
            }
            for doc in docs
        ]

    prompt = f"""You are a helpful assistant answering questions about a YouTube video based on its transcript.
    Use the provided transcript context to answer. If the answer is clearly not in the context, say so briefly —
    but don't refuse if the context contains relevant information, even partially.

    Transcript context:
    {context}

    Question: {question}

    Answer:"""

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": sources
    }