from fastapi import FastAPI
from fastapi.responses import JSONResponse
from vercel import VercelASGI

app = FastAPI()

@app.get("/")
async def root():
    return JSONResponse({"message": "Hello from FastAPI on Vercel!"})

# Wrap FastAPI app for Vercel
handler = VercelASGI(app)
