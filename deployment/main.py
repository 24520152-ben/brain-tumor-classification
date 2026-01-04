import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from router import router as api_router
import os
from config import CURRENT_DIR

app = FastAPI(title="Brain Tumor Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

@app.get("/")
async def read_index():
    index_path = os.path.join(CURRENT_DIR, "index.html")
    return FileResponse(index_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)