import yaml
import cv2
import numpy as np
import requests
import urllib.parse
import logging
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from requests.auth import HTTPDigestAuth
from fastapi import APIRouter, UploadFile, File, Query

from ...src.func.load_model import LoadModel

# ======================= CONFIGURE LOGGING =======================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/app.log",  # Log file
    level=logging.INFO,           # Log level INFO and above
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

# Load configuration
with open("cfg/cfg_base.yaml", 'r') as stream:
    cfg_base = yaml.safe_load(stream)
cfg_base['SAVE_PATH'] = os.getenv("SAVE_PATH")
# Load model
models = LoadModel(cfg_base)

router = APIRouter()

# ======================= HANDLE IMAGE UPLOAD =======================
@router.post("/upload")
async def uploadImage(file: UploadFile = File(...)):
    try:
        logging.info(f"Received uploaded image: {file.filename}")
        
        timestamp = int(time.time())
        date_time = datetime.fromtimestamp(timestamp)
        year = date_time.year
        month = date_time.month
        day = date_time.day
        
        # Create storage directory    
        image_save = "data"
        image_save = f"{image_save}/{year}{month:02d}{day:02d}"
        os.makedirs(image_save, exist_ok=True)

        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            logging.error("Cannot read image")
            return {"Succeeded": False, "Message": "Error cannot read image", "Data": ""}

        input_image_path = os.path.join(image_save, f"{timestamp}_img.jpg")
        cv2.imwrite(input_image_path, img)

        # LP detector
        image_lp = models.lp_detector.predict_image(img)

        if image_lp is None:
            logging.warning("License plate not detected")
            return {"Succeeded": False, "Message": "License plate cannot be detected", "Data": {"timestamp": timestamp, "img_path": f"{cfg_base['SAVE_PATH']}/{'/'.join(input_image_path.split('/')[1:])}"}} 

        # LP recognition
        number_lp, conf_lp = models.lp_recognition.predict_image(image_lp, 0)

        lp_image_path = os.path.join(image_save, f"{timestamp}_lp_{number_lp}.jpg")
        cv2.imwrite(lp_image_path, image_lp)

        if number_lp == "":
            logging.warning("License plate recognition failed")
            return {"Succeeded": False, "Message": "License plate recognition failed", "Data": {"timestamp": timestamp, "img_path": f"{cfg_base['SAVE_PATH']}/{'/'.join(input_image_path.split('/')[1:])}", "lp_path": f"{cfg_base['SAVE_PATH']}/{'/'.join(lp_image_path.split('/')[1:])}"}}
        
        result = {
            "Succeeded": True,
            "Message": "License plate recognition succeeded",
            "Data": {
                "timestamp": timestamp,
                "img_path": f"{cfg_base['SAVE_PATH']}/{'/'.join(input_image_path.split('/')[1:])}",
                "lp_path": f"{cfg_base['SAVE_PATH']}/{'/'.join(lp_image_path.split('/')[1:])}",
                "number": number_lp,
                "score": round(conf_lp, 5)
            }
        }

        logging.info(f"License plate recognized successfully: {result['Data']}")
        return result

    except Exception as e:
        logging.exception("Error processing uploaded image")
        return {"Succeeded": False, "Message": "Error cannot read image", "Data": ""}


# ======================= HANDLE SNAPSHOT =======================
@router.post("/url")
async def uploadURL(url: str = Query(..., description="Image URL (Snapshot or Web)")):
    try:
        decoded_url = urllib.parse.unquote(url)
        logging.info(f"Received image URL: {decoded_url}")

        # Get timestamp & date for saving images
        timestamp = int(time.time())
        date_time = datetime.fromtimestamp(timestamp)
        year = date_time.year
        month = f"{date_time.month:02d}"
        day = f"{date_time.day:02d}"

        # Create storage directory    
        image_save = f"data/{year}{month}{day}"
        os.makedirs(image_save, exist_ok=True)

        # Parse URL to check for authentication
        parsed = urllib.parse.urlparse(decoded_url)
        username, password = parsed.username, parsed.password

        # Download image
        auth = HTTPDigestAuth(username, password) if username and password else None
        response = requests.get(decoded_url, auth=auth, timeout=5)

        if response.status_code != 200:
            logging.error(f"Error downloading image, status: {response.status_code}")
            return {"Succeeded": False, "Message": "Error reading image from URL", "Data": ""}

        # Ensure response is an image
        content_type = response.headers.get("Content-Type", "")
        if "image" not in content_type:
            logging.error(f"Invalid content type: {content_type}")
            return {"Succeeded": False, "Message": "URL does not point to an image", "Data": ""}

        # Decode image
        image_np = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if img is None:
            logging.error("Cannot decode image from URL")
            return {"Succeeded": False, "Message": "Error cannot read image", "Data": ""}

        # Save input image
        input_image_path = os.path.join(image_save, f"{timestamp}_img.jpg")
        cv2.imwrite(input_image_path, img)

        # License Plate Detection
        image_lp = models.lp_detector.predict_image(img)
        if image_lp is None:
            logging.warning("License plate not detected")
            return {
                "Succeeded": False,
                "Message": "License plate cannot be detected",
                "Data": {"timestamp": timestamp, "img_path": f"{cfg_base['SAVE_PATH']}/{'/'.join(input_image_path.split('/')[1:])}"}
            } 

        # License Plate Recognition
        number_lp, conf_lp = models.lp_recognition.predict_image(image_lp, 0)

        # Save LP image
        lp_image_path = os.path.join(image_save, f"{timestamp}_lp_{number_lp}.jpg")
        cv2.imwrite(lp_image_path, image_lp)

        if number_lp == "":
            logging.warning("License plate recognition failed")
            return {
                "Succeeded": False,
                "Message": "License plate recognition failed",
                "Data": {
                    "timestamp": timestamp,
                    "img_path": f"{cfg_base['SAVE_PATH']}/{'/'.join(input_image_path.split('/')[1:])}",
                    "lp_path": f"{cfg_base['SAVE_PATH']}/{'/'.join(lp_image_path.split('/')[1:])}"
                }
            }

        # Success response
        result = {
            "Succeeded": True,
            "Message": "License plate recognition succeeded",
            "Data": {
                "timestamp": timestamp,
                "img_path": f"{cfg_base['SAVE_PATH']}/{'/'.join(input_image_path.split('/')[1:])}",
                "lp_path": f"{cfg_base['SAVE_PATH']}/{'/'.join(lp_image_path.split('/')[1:])}",
                "number": number_lp,
                "score": round(conf_lp, 5)
            }
        }

        logging.info(f"License plate recognized successfully: {result['Data']}")
        return result

    except Exception as e:
        logging.exception("Error processing image from URL")
        return {"Succeeded": False, "Message": "Error cannot read image", "Data": ""}
