import json
from typing import Optional
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import os, uuid, datetime, asyncio, logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import torch
from io import BytesIO
import moviepy.editor as mp
import numpy as np
import motor.motor_asyncio # MongoDB async driver
from kafka import KafkaProducer

from server_block import gradio_demo, api_demo ##, cli_demo

logger.warning("Starting FastAPI app")
app = FastAPI()


MONGO_DETAILS = os.getenv("MONGO_URI", "mongodb://localhost:27017")
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
VIDEO_STORAGE_DIR = "generated_videos" # Directory to save videos
os.makedirs(VIDEO_STORAGE_DIR, exist_ok=True) # Create if not exists
# --- MongoDB Client ---
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)
database = client.video_app_db #  database name
jobs_collection = database.jobs # collection for video generation jobs
# --- Kafka Producer (initialized on startup) ---
kafka_producer = None

@app.get("startup")
async def startup_db_client():
    global kafka_producer
    logger.info(f"Connecting to MongoDB at {MONGO_DETAILS}")
    try:
        # The connection is lazy, try to ping a command to ensure connectivity
        await client.admin.command('ping')
        logger.info("MongoDB connection successful!")
    except Exception as e:
        # Optionally, raise an exception here to prevent startup
        logger.error(f"MongoDB connection failed: {e}")

    logger.info(f"Connecting to Kafka broker at {KAFKA_BROKER}")
    try:
        kafka_producer = KafkaProducer(
            bootstrap_servers=KAFKA_BROKER.split(','),
            value_serializer=lambda v: v.encode('utf-8'), acks='all') 
        # Ensure message is received by leader and followers
        logger.info("Kafka producer connected successfully!")
    except Exception as e:
        logger.error(f"Kafka producer connection failed: {e}")
        # Optionally, raise an exception or handle this more robustly


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
    if kafka_producer:
        kafka_producer.close()
        logger.info("Kafka producer closed.")
    logger.info("MongoDB client closed.")

# --- Request Models ---
class VideoRequest(BaseModel): # or we can name it TextPrompt 
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
    message: str = None # For error messages or additional info



@app.post('/generate-video', status_code=202) # 202 Accepted
async def generate_video_request(request: VideoRequest):
    if kafka_producer is None:
        raise HTTPException(
                status_code=503, 
                detail="Kafka producer not initialized. Service unavailable.")

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
        "message": "Job received and queued."
    }

    try:
        await jobs_collection.insert_one(job_data)
        logger.info(f"Job {job_id} stored in MongoDB.")

        # Publish job ID to Kafka
        kafka_producer.send('video_generation_requests', value=job_id)
        kafka_producer.flush() # Ensure message is sent

        logger.info(f"Job {job_id} sent to Kafka topic 'video_generation_requests'.")

        return {"job_id": job_id, 
                "status": "PENDING", 
                "message": "Video generation request accepted and queued."}
          
    except Exception as e:
        logger.error(f"Error queuing video generation job {job_id}: {e}")
        raise HTTPException(status_code=500, 
                            detail=f"Failed to queue video generation job: {e}")


@app.get('/download-video/{job_id}')
async def download_video(job_id: str):
    job = await jobs_collection.find_one({"job_id": job_id})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    if job.get("status") != "COMPLETED":
        raise HTTPException(status_code=409,
                            detail=f"Video for job {job_id} is not yet completed. \
                                     Current status: {job.get('status')}")

    video_path = job.get("video_path")
    if not video_path or not os.path.exists(video_path):
        logger.error(f"Video file not found for job {job_id} at {video_path}")
        raise HTTPException(status_code=500,
                            detail="Video file not found or path invalid.")

    logger.info(f"Streaming video for job {job_id} from {video_path}")
    return FileResponse(path=video_path,
                        media_type="video/mp4",
                        filename=f"generated_video_{job_id}.mp4")

@app.route("/health")
async def health():
    return {"success": True}, 200 

@app.get('/')
async def root():
    return {'message': 'Text to video API is running. Use /generate-video to start a job.'} 

