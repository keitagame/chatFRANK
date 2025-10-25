from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Hugging Face のパイプラインをロード
generator = pipeline("text-generation", model="gpt2")

# FastAPI アプリ作成
app = FastAPI()

# 入力データの型定義
class Prompt(BaseModel):
    text: str

@app.post("/generate")
def generate_text(prompt: Prompt):
    result = generator(prompt.text, max_length=50, num_return_sequences=1)
    return {"output": result[0]["generated_text"]}
