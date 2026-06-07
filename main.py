from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # ← thêm
from fastapi.responses import FileResponse          # ← thêm
from pydantic import BaseModel
import numpy as np
import traceback  # ← thêm dòng này
from MoveFile import moveFile
from ModelLoad import modelLoad
import sys

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
class LoginData(BaseModel):
    username: str

current_username = None
@app.get("/")
async def index():
    return FileResponse("index.html")

@app.post('/get-username')
async def login(loginData: LoginData):
    global current_username
    try:
        current_username = loginData.username
        moveFile(current_username)
        return {'message': 'Data has implemented'}
    except Exception as e:
        traceback.print_exc()  # ← thêm dòng này
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data-label")
async def getData():
    try:
        segmented_data, labels, prediction_probability = modelLoad(current_username)
        segmented_data = segmented_data.tolist()
        labels = labels.tolist()
        prediction_probability = prediction_probability.tolist()
        return {'data': segmented_data, 'label': labels, 'prediction_probability': prediction_probability}
    except Exception as e:
        traceback.print_exc()  # ← thêm dòng này
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True, timeout_keep_alive=60)