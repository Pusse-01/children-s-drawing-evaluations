from fastapi import FastAPI, File, UploadFile
import uvicorn
from src.utils import Model
import json



app = FastAPI()

#root
@app.get("/")
def root():
    return "api running!"



#dialog bulk messages
@app.post("/check-difference")
async def check(source_file: UploadFile = File(...),drawing_file: UploadFile = File(...)):

    model=Model()
    percentage=model.predict(source_file=source_file,drawing_file=drawing_file)

    # res = im.fromarray(res)
    # res.save('result.png')
    
    return percentage
















if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)