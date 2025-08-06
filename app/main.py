from fastapi import FastAPI, File, UploadFile
from app.model import predict_animal

app = FastAPI(title="Animal Detection API")

@app.get("/")
def root():
    return {"status": "API funcionando correctamente 🚀"}

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    result = predict_animal(contents)
    return result
