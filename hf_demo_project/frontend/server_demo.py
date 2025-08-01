from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
##from motor.motor_asyncio import AsyncIOMotorClient
#from kafka import KafkaProducer
import json
from typing import Optional
import uuid

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from server_block import gradio_demo, api_demo ##, cli_demo

gradio_ui = gradio_demo()
gradio_ui.queue()

logger.warning("Starting FastAPI app")
app = FastAPI()

app = gr.mount_gradio_app(app, gradio_ui, '/')

@app.post('/generate-video')
async def generate_video(request: t_prompt):
    return gradio_demo() 

@app.route("/health")
async def health():
    return {"success": True}, 200 

@app.get('/')
async def root():
    return {'message': 'Text to video API is running. Use /generate-video to start a job.'} 
