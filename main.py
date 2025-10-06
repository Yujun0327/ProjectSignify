from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import base64
import numpy as np
import cv2
import logging

from utils import *

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up logging
logging.basicConfig(level=logging.INFO)

class ImageData(BaseModel):
    image: str

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/home.html") as f:
        return f.read()

class TextData(BaseModel):
    text: str

# Define the FastAPI endpoint to correct the text
@app.post("/api/correct")
async def correct(text_data: TextData):
    try:
        # Call the string correction function
        corrected_text = aslstringcorrection(text_data.text)
        print("CORRECTED TEXT: %s" % corrected_text)

        tts(corrected_text)
        return {"corrected_text": corrected_text}
    
    except Exception as e:
        # Handle exceptions and send an appropriate HTTP response
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/api/stt")
async def stt(text_data: TextData):
    print("STT: %s" % text_data)
    try:
        # Call the string correction function
        # Example usage
        record_audio(3)  # Record 5 seconds of audio
        recognized_text = recognize_speech()
        
        make_png(recognized_text) #combined_image.png
        image_path = "combined_image.png"  # 이미지 파일의 경로

        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

        print("보내기")
         # JSON 형식으로 이미지와 텍스트를 함께 반환
        return JSONResponse(content={"image_data": encoded_string, "text_data": recognized_text})


    except Exception as e:
        # Handle exceptions and send an appropriate HTTP response  
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/api/translate")
async def translate(image_data: ImageData):
    try:
        encoded = image_data.image
        logging.info(f"Received encoded image data length: {len(encoded)}")

        # Decode base64 to an image
        try:
            image = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        except Exception as decode_error:
            logging.error(f"Decoding error: {decode_error}")
            raise HTTPException(status_code=400, detail="Invalid base64 image data")

        # Check if image decoding was successful
        if image is None:
            logging.error("Image decoding resulted in None")
            raise HTTPException(status_code=400, detail="Image decoding failed")

        # Classify the image using the function from utils.py
        predicted_class = classify_image(image)

        return JSONResponse(content={"predicted_class": predicted_class, "confidence": 100})

    except Exception as e:
        # Log the error for debugging purposes
        logging.error(f"Error in /api/translate: {str(e)}")
        raise HTTPException(status_code=400, detail="An error occurred during translation")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
