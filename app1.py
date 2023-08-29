from PIL import Image
import io  # provides tools for working with streams of binary data.
import pandas as pd  # used for data manipulation and analysis.
import numpy as np  # used for numerical computations and array operations.
from typing import Optional
from ultralytics import YOLO
# from ultralytics.yolo.utils.plotting import Annotator, colors
# from ultralytics.data.annotator import auto_annotate
import cv2 # used for computer vision tasks such as image and video processing.
from fastapi import FastAPI, File, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException
import logger
from loguru import logger
import json
import sys  # Imports the built-in module for interpreter-related interactions and variables.

app = FastAPI(
    title="Yolov8 Garbage Detection FastAPI ",
    description= """ Takes Input image and returns the detected objects image and json object""",
    version="28-08-2023",
)

origins = [
    "http://localhost",
    "http://localhost:8008",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
    level=10,
)
logger.add("log.log", rotation="1 MB", level="DEBUG", compression="zip")

best_model = YOLO("./best.pt")

def image_from_bytes(binary_image: bytes) -> Image:

    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    return input_image

def bytes_from_image(image: Image) -> bytes:

    bytes_image = io.BytesIO()
    image.save(bytes_image, format='JPEG', quality=85)  # save the image in JPEG format with quality 85
    bytes_image.seek(0)  # set the pointer to the beginning of the file
    return bytes_image

def Prediction_to_dataframe(results: list, labeles_dict: dict) -> pd.DataFrame:
    # Transform the Tensor to numpy array
    
    predict_bounding_box = pd.DataFrame(results[0].to("cpu").numpy().boxes.xyxy, columns=['xmin', 'ymin', 'xmax', 'ymax'])
    # Add the confidence of the prediction to the DataFrame
    predict_bounding_box['confidence'] = results[0].to("cpu").numpy().boxes.conf
    # Add the class of the prediction to the DataFrame
    predict_bounding_box['class'] = (results[0].to("cpu").numpy().boxes.cls).astype(int)
    # Replace the class number with the class name from the labeles_dict
    predict_bounding_box['name'] = predict_bounding_box["class"].replace(labeles_dict)
    return predict_bounding_box


def get_model_predict(model: YOLO, input_image: Image, save: bool = False, image_size: int = 640, conf: float = 0.5, augment: bool = False) -> pd.DataFrame:
    predictions = model.predict(
        imgsz=image_size,
        source=input_image,
        conf=conf,
        save=save,
        augment=augment,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
    )

    # predictions to pandas dataframe conversion
    predictions = Prediction_to_dataframe(predictions, model.model.names)
    return predictions

def add_bboxs_on_img(image: Image, predict: pd.DataFrame) -> Image:
    # Convert PIL Image to NumPy array
    image_np = np.array(image)

    # Iterate over the rows of the predict DataFrame
    for i, row in predict.iterrows():
        # Get bounding box coordinates
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

        # Draw the bounding box rectangle on the image using OpenCV
        color = (255, 255, 255)  # Color of the bounding box
        thickness = 4  # Thickness of the bounding box
        cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), color, thickness)

        # Add text label with class name and confidence
        label = f"{row['name']}: {int(row['confidence'] * 100)}%"
        cv2.putText(image_np, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, thickness)

    # Convert the NumPy array back to a PIL Image
    final_image = Image.fromarray(image_np)
    return final_image

def garbage_model_detection(input_image: Image) -> pd.DataFrame:

    predict = get_model_predict(
        model=best_model,
        input_image=input_image,
        save=False,
        image_size=640,
        augment=False,
        conf=0.5,
    )
    return predict

@app.on_event("startup")
def save_openapi_json():
    openapi_data = app.openapi()
    # Change "openapi.json" to desired filename
    with open("openapi.json", "w") as file:
        json.dump(openapi_data, file)

@app.get('/LifeCheck', status_code=status.HTTP_200_OK)
def perform_LifeCheck():
    return {'LifeCheck': 'Yeah Connection ALIVE....Not Dead Yet !'}


@app.post("/object_detection_to_json")
def object_detection_to_json(file: bytes = File(...)):
    result = {'detect_objects': None}

    # # Step 2: Convert the image file to an image object
    # input_image = image_from_bytes(file)

    # Step 3: Predict from model
    predict = garbage_model_detection(image_from_bytes(file))

    # Step 4: Select detect obj return info - here you can choose what data to send to the result
    detect_res = predict[['name', 'confidence']]
    objects = detect_res['name'].values

    result['detect_objects_names'] = ', '.join(objects)
    result['detect_objects'] = json.loads(detect_res.to_json(orient='records'))

    # Step 5: Logs and return
    logger.info("results: {}", result)
    return result


@app.post("/object_detection_to_img")
def object_detection_to_img(file: bytes = File(...)):
    # get image from bytes
    input_image = image_from_bytes(file)

    # model predict
    predict = garbage_model_detection(input_image)

    # add bbox on image
    final_image = add_bboxs_on_img(image=input_image, predict=predict)

    # return image in bytes format
    return StreamingResponse(content=bytes_from_image(final_image), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
