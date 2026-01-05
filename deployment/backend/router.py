from fastapi import APIRouter, File, UploadFile, HTTPException
from service import model
import traceback

router = APIRouter()

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        return model.predict_and_explain(contents)
    except Exception as e:
        traceback.print_exc() # In lỗi chi tiết ra Terminal để debug
        raise HTTPException(status_code=500, detail=str(e))