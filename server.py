import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import tensorflow as tf
import numpy as np

app = FastAPI()

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pickle', 'rb') as ecn_file:
    lbl_encoder = pickle.load(ecn_file)

model = tf.keras.models.load_model('chat_model.h5')

class MessageInput(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "Welcome to the NLP classification API."}

@app.post("/classify")
async def classify_message(message: MessageInput):
    print(f"Received message: {message.text}")
    try:
        sequence = tokenizer.texts_to_sequences([message.text])
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, truncating='post', maxlen=20)
        prediction = model.predict(padded_sequence)
        class_index = np.argmax(prediction[0])
        tag = lbl_encoder.inverse_transform([class_index])[0]
        print(f"Classification result: {tag}")
        return {"classification": tag, "input_text": message.text}
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="192.168.193.30", port=8081)
