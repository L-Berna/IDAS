# To run this, you need to install:
# pip install openai
# pip install elevenlabs
# pip install SpeechRecognition
# pip install pydub

# Standard libraries
import os
import sys
import asyncio
import tempfile
import requests
import keyboard
import warnings

# RAG libraries
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai import OpenAIEmbeddings
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Speech-related libraries
import pydub
from pydub import playback
import speech_recognition as sr
from elevenlabs import generate, play, set_api_key, voices, Models

openai_api_key = os.environ.get('OPENAI_API_KEY')
llm_api_key = os.environ.get('OPENAI_API_KEY')
eleven_api_key = os.environ.get('ELEVEN_LABS_KEY')
output_wav = os.path.join(os.getcwd(), "audio.wav")

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

recognizer = sr.Recognizer()

# Configure OpenAI and Text-to-speech API keys
openai_client = OpenAI(api_key=openai_api_key)
set_api_key(eleven_api_key)

# Configure Embedding model
embedding_dimensions = 3072
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=embedding_dimensions)

# Vector dataset
vectordb_directory = os.path.join(os.getcwd(), f'vector_database_chspark_{embedding_dimensions}')
if not os.path.exists(vectordb_directory):
    print("No vector database found")
    print(vectordb_directory)

print(f"Using vector database {vectordb_directory}")

# Create chroma db from existing vectordb_directory
vectordb = Chroma(
    embedding_function=embedding_model,
    persist_directory=vectordb_directory
)

print(f"Load {vectordb._collection.count()} collections from vector database")

# Configure voice
voice_list = voices()
voice_labels = [voice.category + " voice: " + voice.name for voice in voice_list]
#print(voice_labels)

# Select voice to use
voice_id = "generated voice: Saci2"
#voice_id = "cloned voice: Juan"  
selected_voice_index = voice_labels.index(voice_id)
selected_voice_id    = voice_list[selected_voice_index].voice_id

# Configure ChatGPT.
llm_name = "gpt-4o" #"gpt-3.5-turbo" #"gpt-4"

template = """\
You are a car assistant called IDAS that stands for Intelligent Driving Assistance System. 
Your job is to answer the questions that are asked to you. 
If the question is of general knowledge, respond with what you know.
You are an intelligent assistant on board of a Chevrolet Spark 2020.
If the question is specific about the vehicle, use the provided context information taken from the car owners manual to answer the question at the end. 
If you don’t know the answer about the vehicle even with context information provided say "I am sorry, I did not find the answer in the car manual"
Don’t try to make up an answer.
Respond in the most attentive way possible. Use a maximum of two sentences. 
Keep the answer as concise as possible. 
Respond in the same language as the question.
Context: {context}
Question: {question}
Helpful Answer:
"""

def play_audio(file):
    sound = pydub.AudioSegment.from_file(file, format="mp3")
    playback.play(sound)


def text_to_voice(text):
    CHUNK_SIZE = 1024
    url = "https://api.elevenlabs.io/v1/text-to-speech/" + selected_voice_id

    headers = {
      "Accept": "audio/mpeg",
      "Content-Type": "application/json",
      "xi-api-key": eleven_api_key
    }

    data = {
      "text": text,
      "model_id" : "eleven_multilingual_v1",
      "voice_settings": {
        "stability": 0.4,
        "similarity_boost": 1.0
      }
    }

    response = requests.post(url, json=data, headers=headers)
    
    # Save audio data to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
        f.flush()
        temp_filename = f.name

    return temp_filename

    
# Function to continuously interact with GPT
async def main(qa_chain):
    while True:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            print("\nEsperando por la tecla espacio para escucharte ...")

            # Wait until spacebar or Escape
            while True:
                if keyboard.is_pressed('space'):
                    break
                elif keyboard.is_pressed('esc'):
                    print("Hasta pronto!")
                    sys.exit(0)

            print("\nTe escucho ...")
            audio = recognizer.listen(source)

            try:
                with open("audio_prompt.wav", "wb") as f:
                    f.write(audio.get_wav_data())
                
                audio_file= open("audio_prompt.wav", "rb")
                result = openai_client.audio.transcriptions.create(model="whisper-1", file=audio_file)
                audio_file.close()
                query = result.text
                print(f"Dijiste: {query}")
            except Exception as e:
                print(f"Error transcribiendo audio: {e}")
                continue

        # User command to GPT
        model_response = qa_chain.invoke({"query": query})
        response_text = model_response["result"]
        print("IDAS:", response_text)

        # Text response to audio
        audio_file = text_to_voice(response_text)
        play_audio(audio_file)

if __name__ == "__main__":
    # create prompt template object
    qa_chain_prompt = PromptTemplate.from_template(template)

    if "gpt" in llm_name:
        llm = ChatOpenAI(model_name=llm_name, api_key=llm_api_key, temperature=0) 
    elif "claude" in llm_name:    
        llm = ChatAnthropic(model_name=llm_name, api_key=llm_api_key, temperature=0)
    elif "mistral" in llm_name or "mixtral" in llm_name:
        llm = ChatMistralAI(model=llm_name, api_key=llm_api_key, temperature=0)
    elif "llama" in llm_name or "gemma" in llm_name:
        llm = ChatOpenAI(model_name=llm_name, api_key=llm_api_key, temperature=0,
                        base_url="https://api.llama-api.com") 
    print(f"Using Model: {llm.model_name}")

    # QA RAG object
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_chain_prompt}
    )
    asyncio.run(main(qa_chain))

