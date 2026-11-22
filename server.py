import os, re, threading, uuid
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import clipper

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)
jobs = {}
app = FastAPI()


class Req(BaseModel):
    url: str
    length: int = 35
    count: int = 5


def run(job_id, req):
    job = jobs[job_id]
    folder = os.path.join(OUT, job_id)
    os.makedirs(folder, exist_ok=True)
    try:
        clips = clipper.make_clips(req.url, folder, req.length, req.count,
                                   lambda m, p: job.update(step=m, pct=p))
        job.update(status="done", pct=100, step="Done", clips=clips)
    except Exception as e:
        job.update(status="error", error=str(e).splitlines()[0][:300] if str(e) else "Something went wrong.")


@app.post("/api/clip")
def start(req: Req):
    if not re.match(r"^https?://(www\.|m\.)?(youtube\.com|youtu\.be)/", req.url.strip()):
        raise HTTPException(400, "Paste a full YouTube link.")
    req.length = max(15, min(60, req.length))
    req.count = max(1, min(10, req.count))
    job_id = uuid.uuid4().hex[:10]
    jobs[job_id] = {"status": "working", "step": "Starting", "pct": 2}
    threading.Thread(target=run, args=(job_id, req), daemon=True).start()
    return {"id": job_id}


@app.get("/api/job/{job_id}")
def job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Unknown job.")
    return jobs[job_id]


app.mount("/output", StaticFiles(directory=OUT), name="output")
@app.get("/")
def home():
    return FileResponse(os.path.join(BASE, "index.html"))
