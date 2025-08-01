from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
##from motor.motor_asyncio import AsyncIOMotorClient
#from kafka import KafkaProducer
import json
from typing import Optional
import uuid

## FastAPI with Gradio
import os
import uuid
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, Form
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import torch
from io import BytesIO
import moviepy.editor as mp
import numpy as np
import asyncio
import logging
import motor.motor_asyncio # MongoDB async driver
import traceback # For detailed error logging
import gradio as gr # Import Gradio

# --- Configuration ---
#from dotenv import load_dotenv
#load_dotenv()

MONGO_DETAILS = os.getenv("MONGO_URI", "mongodb://localhost:27017")
VIDEO_STORAGE_DIR = "generated_videos"
os.makedirs(VIDEO_STORAGE_DIR, exist_ok=True)
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from server_block import gradio_demo, api_demo ##, cli_demo

logger.warning("Starting FastAPI app")


# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(title='Video Generation API with Gradio UI')

# --- MongoDB Client ---
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)
database = client.video_app_db
jobs_collection = database.jobs


@app.get("startup")
async def startup_event():
    global pipeline_
    logger.info(f"Connecting to MongoDB at {MONGO_DETAILS}")
    try:
        await client.admin.command('ping')
        logger.info("MongoDB connection successful!")
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        # Consider raising an exception if MongoDB is critical

# --- Request Models (for FastAPI native endpoint) ---
class VideoRequest(BaseModel):
    prompt: str
    negative_prompt: str = None
    num_frames: int = 16
    num_inference_steps: int = 84
    guidance_scale: float = 7.5
    fps: int = 8

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    video_path: str = None
    created_at: datetime
    updated_at: datetime
    message: str = None


# --- Background Task Function for Video Generation ---
async def _generate_video_task(job_id: str, request_data: dict):
    """
    This function performs the actual video generation.
    It runs in a background task (separate thread) managed by FastAPI/ASGI server.
    """
    await jobs_collection.update_one(
        {"job_id": job_id},
        {"$set":
              {"status": "PROCESSING",
               "updated_at": datetime.utcnow(),
               "message": "Video generation started."}}
    )
    logger.info(f"Background task: Started processing job {job_id}.")

    try:
        gradio_demo()
        logger.info(f"Background task: Video for job {job_id} has been saved")

        await jobs_collection.update_one(
            {"job_id": job_id},
            {"$set": {
                "status": "COMPLETED",
                "video_path": '/models/outputs',
                "updated_at": datetime.utcnow(),
                "message": "Video generation completed successfully."
            }}
        )
        logger.info(f"Background task: Job {job_id} status updated to COMPLETED in MongoDB.")

    except Exception as e:
        logger.error(f"Background task: Error generating video for job {job_id}: {e}", exc_info=True)
        await jobs_collection.update_one(
            {"job_id": job_id},
            {"$set": {
                "status": "FAILED",
                "updated_at": datetime.utcnow(),
                "message": f"Video generation failed: {e}"
            }}
        )
        logger.info(f"Background task: Job {job_id} status updated to FAILED in MongoDB.")


@app.post('/generate-video', status_code=202)
async def generate_video_request_api(
    request: VideoRequest, background_tasks: BackgroundTasks):
    if pipeline_ is None:
        raise HTTPException(status_code=503, 
          detail="Video generation model is not loaded. Service unavailable.")

    job_id = str(uuid.uuid4())
    job_data = {
        "job_id": job_id,
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "num_frames": request.num_frames,
        "num_inference_steps": request.num_inference_steps,
        "guidance_scale": request.guidance_scale,
        "fps": request.fps,
        "status": "PENDING",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "video_path": None,
        "message": "Job received and queued for background processing."
    }

    try:
        await jobs_collection.insert_one(job_data)
        logger.info(f"API: Job {job_id} stored in MongoDB and queued for background generation.")
        background_tasks.add_task(_generate_video_task, job_id, request.dict())
        return {"job_id": job_id, 
                "status": "PENDING", 
                "message": "Video generation request accepted and processing in background."}
    except Exception as e:
        logger.error(f"API: Error queuing video generation job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to queue video generation job: {e}")



@app.get('/get-video-status/{job_id}', response_model=JobStatusResponse)
async def get_video_status(job_id: str):
    job = await jobs_collection.find_one({"job_id": job_id})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    job['job_id'] = job_id
    job['created_at'] = job['created_at'].isoformat()
    job['updated_at'] = job['updated_at'].isoformat()

    return JobStatusResponse(**job)

@app.get('/download-video/{job_id}')
async def download_video(job_id: str):
    job = await jobs_collection.find_one({"job_id": job_id})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    if job.get("status") != "COMPLETED":
        if job.get("status") == "PENDING":
             detail_message = f"Video for job {job_id} is pending. Please wait."
        elif job.get("status") == "PROCESSING":
             detail_message = f"Video for job {job_id} is still being processed. Current status: {job.get('status')}"
        elif job.get("status") == "FAILED":
             detail_message = f"Video generation for job {job_id} failed. Reason: {job.get('message', 'Unknown error.')}" 
        else:
             detail_message = f"Video for job {job_id} is not yet completed. Current status: {job.get('status')}"
        raise HTTPException(status_code=409, detail=detail_message)

    video_path = job.get("video_path")
    if not video_path or not os.path.exists(video_path):
        logger.error(f"Video file not found for job {job_id} at {video_path}")
        raise HTTPException(status_code=500, detail="Video file not found or path invalid.")

    logger.info(f"Streaming video for job {job_id} from {video_path}")
    return FileResponse(path=video_path, media_type="video/mp4", filename=f"generated_video_{job_id}.mp4")

