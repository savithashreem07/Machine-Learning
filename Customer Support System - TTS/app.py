from flask import Flask, request, send_file, render_template
import os
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
import queue
import threading
import time
from pydub import AudioSegment
from pydub.playback import play
import speech_recognition as sr
from gtts import gTTS
import numpy as np
import torch
import re
import whisper
from openai import OpenAI

app = Flask(__name__)

with open("/home/savitha07/.env") as env:
    for line in env:
        key, value = line.strip().split('=')
        os.environ[key] = value

client = OpenAI(
    api_key=os.environ.get('OPENAI_API_KEY'),
)


# Initialize LangChain components
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0301", temperature=0.1)
embeddings = OpenAIEmbeddings()
loader = PyPDFLoader("2023Catalog.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
documents = text_splitter.split_documents(docs)
vector = DocArrayInMemorySearch.from_documents(documents, embeddings)
retriever = vector.as_retriever()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

audio_model = whisper.load_model("base")

def listen_and_transcribe():
    r = sr.Recognizer()
    r.energy_threshold = 300
    r.pause_threshold = 0.8
    r.dynamic_energy_threshold = False

    with sr.Microphone(sample_rate=16000) as source:
        print("Listening...")
        audio = r.listen(source)
        torch_audio = torch.from_numpy(np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
        result = audio_model.transcribe(torch_audio, language='english')
        return result["text"]


def get_completion_from_messages(messages, 
                                 model="gpt-3.5-turbo", 
                                 temperature=0, 
                                 max_tokens=150):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature, 
        max_tokens=max_tokens, 
    )
    print(response.choices[0])
    return response.choices[0].message.content


def main():
    while True:
        user_input = listen_and_transcribe()
        if user_input.lower() == "quit":
            break
        result = qa.invoke({"question": user_input})
        # response_text = result['answer']
        # print("Response:", response_text)
        # synthesize_speech(response_text)

        messages = [{"role": "user", "content": result}]
        answer = get_completion_from_messages(messages)
        
        mp3_obj = gTTS(text=answer, lang="en", slow=False)
        mp3_obj.save("reply.mp3")
        reply_audio = AudioSegment.from_mp3("reply.mp3")
        print("Playing audio..")
        play(reply_audio)
        os.remove("reply.mp3")

if __name__ == "__main__":
    main()

