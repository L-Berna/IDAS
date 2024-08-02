
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import re
import asyncio
import threading
import configparser
import argparse

from openai import OpenAI
import pyaudio
import wave
import platform
from ctypes import *
from langchain_anthropic import ChatAnthropic
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from colorama import init, Fore
import tkinter as tk
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
import logging

from rag_utils import get_session_history, contextualize_q_system_prompt

logging.getLogger().setLevel(logging.ERROR) # hide warning log
os_name = platform.system()  # get the name of the OS
init(autoreset=True) # Initialize colorama

# Audio output streaming needs mpv in Windows
# https://mpv.io/installation/
if os_name == "Windows":
    parent_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    os.environ['PATH'] += os.pathsep + parent_directory + '/mpv/'

# Global Variables
event_loop = None   # store the event loop
app_running = True  # Flag to indicate if the app is running
rec_stream = None   # Audio recording
state = "waiting"   # System state
wav_file = None     # Wave file object
session_id = 1

# Audio file characteristics
temp_file = "prompt_recording.wav" # file to store the prompt audio
sample_rate = 16000
bits_per_sample = 16
chunk_size = 1024
audio_format = pyaudio.paInt16
channels = 1

# Set up argument parser
parser = argparse.ArgumentParser(description='Load configuration for the application.')
parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')

# Parse the arguments
args = parser.parse_args()

# Create a ConfigParser object
config = configparser.ConfigParser()

# Read the configuration file specified by the command-line argument
config.read(args.config)

# Access the values from the config file
vehicle_name = config.get('DEFAULT', 'vehicle_name')
embedding_dimensions = config.getint('DEFAULT', 'embedding_dimensions')
llm_name = config.get('DEFAULT', 'llm_name')
system_prompt = config.get('DEFAULT', 'system_prompt')
vectordb_directory = config.get('DEFAULT', 'vectordb_directory')
voice_name = config.get('DEFAULT', 'voice_name')

# Get the proper LLM API key 
if "gpt" in llm_name:
    llm_api_key = os.environ['OPENAI_API_KEY']
elif "claude" in llm_name:
    llm_api_key = os.environ['ANTHROPIC_API_KEY']
elif "mistral" in llm_name or "mixtral" in llm_name:
    llm_api_key = os.environ['MISTRAL_API_KEY']
elif "llama" in llm_name or "gemma" in llm_name:
    llm_api_key = os.environ['LLAMA_API_KEY']
else:
    print(Fore.RED + "INVALID MODEL!")
    sys.exit(0)

openai_api_key = os.environ.get('OPENAI_API_KEY')
eleven_api_key = os.environ.get('ELEVEN_LABS_KEY')
output_wav = os.path.join(os.getcwd(), "audio.wav")

# Set the LLM
if "gpt" in llm_name:
    llm = ChatOpenAI(model_name=llm_name, api_key=llm_api_key, temperature=0) 
elif "claude" in llm_name:    
    llm = ChatAnthropic(model_name=llm_name, api_key=llm_api_key, temperature=0)
elif "mistral" in llm_name or "mixtral" in llm_name:
    llm = ChatMistralAI(model=llm_name, api_key=llm_api_key, temperature=0)
elif "llama" in llm_name or "gemma" in llm_name:
    llm = ChatOpenAI(model_name=llm_name, api_key=llm_api_key, temperature=0,
                    base_url="https://api.llama-api.com") 
else:
    print(Fore.RED + "UNABLE TO SET THE LLM")
    sys.exit(0)

if hasattr(llm, "model_name"):
    print(Fore.GREEN + f"Using Model: {llm.model_name}")
else:
    print(Fore.GREEN + f"Using Model: {llm_name}")

# Configure OpenAI and Text-to-speech API keys
openai_client = OpenAI(api_key=openai_api_key) # for whisper
elabs_client = ElevenLabs(api_key=eleven_api_key)

# Configure Embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large",
                                   dimensions=embedding_dimensions)

# Vector dataset
if not os.path.exists(vectordb_directory):
    print(Fore.RED + "No vector database found :(")
    print(Fore.RED + vectordb_directory)
    print(Fore.RED + "Exiting!")
    sys.exit(0)
else:
    print(Fore.GREEN + f"Using vector database {vectordb_directory}")

# Data retriever
vectordb = Chroma(
    embedding_function=embedding_model,
    persist_directory=vectordb_directory
)
retriever = vectordb.as_retriever()
print(Fore.GREEN +
      f"Load {vectordb._collection.count()} collections from vector database")

# Contextualize question
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Conversational RAG object
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

conversational_rag_chain = RunnableWithMessageHistory(
rag_chain,
get_session_history,
input_messages_key="input",
history_messages_key="chat_history",
output_messages_key="answer",
)

# Suppress ALSA warnings (https://stackoverflow.com/a/13453192)
if os_name == 'Linux':
    ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
    def py_error_handler(filename, line, function, err, fmt):
        return
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    asound = cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Function to write colored text to the Text widget safely from any thread
def write_text_safe(text, color):
    root.after(0, write_text, text, color)

# Function to write colored text to the Text widget
def write_text(text, color):
    # Define a unique tag name based on the color
    tag_name = f"color_{color}"
    
    # Check if the tag is already defined, if not, define it
    if tag_name not in text_box.tag_names():
        text_box.tag_configure(tag_name, foreground=color)
    
    # Insert the text into the text widget and apply the tag
    text_box.insert(tk.END, text, tag_name)
    text_box.see(tk.END)  # Scroll to the end

def clear_text_box():
    """Clear the content of the text box."""
    text_box.delete("1.0", tk.END)

def new_session():
    global session_id
    session_id += 1
    clear_text_box()

async def process_audio():
    global session_id
    try:
        # Open file and send to whisper (speech to text)
        audio_file = open(temp_file, "rb")
        query_transcription = openai_client.audio.transcriptions.create(
            model="whisper-1", file=audio_file)
        query = query_transcription.text
        print(Fore.CYAN + f"\nYou said: {query}")
        write_text_safe(f"You said: {query}\n", "blue")

        # User command to RAG pipeline
        idas_response = conversational_rag_chain.invoke(
            {"input": query},
            config={
                "configurable": {"session_id": str(session_id)}
            },  # constructs a key "abc123" in `store`.
        )
        response_text = idas_response["answer"]

        # Text response to audio
        audio_stream = elabs_client.generate(
            text=text_stream(response_text),
            voice=voice_name,
            model="eleven_multilingual_v2",
            stream=True
        )

        await asyncio.to_thread(stream, audio_stream)
        root.after(0, reset_button)
    except Exception as e:
        print(f"Error: {e}")
        root.after(0, reset_button)
        root.after(0, write_text, "\n", "")
    finally:
        root.after(0, enable_listening_button)

def enable_listening_button():
    btn_listening.config(state=tk.NORMAL)

def reset_button():
    global state
    state = "waiting"        
    btn_listening["text"] = "Start listening"
    btn_listening.config(bg="green", fg="black")
    write_text("\n", "")

# Function for the big button
def toggle_state():
    global rec_stream, state, wav_file

    if state == "waiting":
        # Change text to stop listening
        btn_listening["text"] = "Stop listening"
        btn_listening.config(bg="lightcoral", fg="black")  # Light red background, black text

        state = "listening"
        write_text("Listening...\n", "green")

        # Initialize wave file before starting recording
        initialize_wave_file()
        
        # Start recording audio
        rec_stream = audio.open(format=audio_format,
                            channels=channels,
                            rate=sample_rate,
                            input=True,
                            frames_per_buffer=chunk_size,
                            stream_callback=audio_recording_callback)

    elif state == "listening":
        # Change text to processing
        btn_listening["text"] = "Processing ..."
        btn_listening.config(bg="#a3a300", fg="black")  # Green background, black text
        #btn_listening.config(state=tk.DISABLED)  # Disable the button
        state = "processing"

        # Stop and close the audio stream
        rec_stream.stop_stream()
        rec_stream.close()

        # Close the wave file
        wav_file.close()
        wav_file = None

        # Start processing audio asynchronously
        asyncio.run_coroutine_threadsafe(process_audio(), event_loop)
    elif state == "processing":
        pass


# Function to initialize the wave file
def initialize_wave_file():
    global wav_file
    wav_file = wave.open(temp_file, 'wb')
    wav_file.setnchannels(channels)
    wav_file.setsampwidth(bits_per_sample // 8)
    wav_file.setframerate(sample_rate)

def audio_recording_callback(in_data, frame_count, time_info, status):
    if wav_file is not None:
        wav_file.writeframes(in_data)
    return None, pyaudio.paContinue

def text_stream(text):
    # Split the text by sentence endings, keeping the punctuation
    sentences = re.split(r'(?<=[.!?]) +|\n', text)
    for sentence in sentences:
        print(Fore.YELLOW + sentence)
        write_text(f"{sentence}\n", "green")
        yield sentence

# Called when closing the app
def on_closing():
    global app_running

    app_running = False  # Indicate that the app is closing

    # Check if rec_stream is not None and has a valid stream object before calling is_active
    try:
        if rec_stream and rec_stream._is_running:
            if rec_stream.is_active():
                rec_stream.stop_stream()
            rec_stream.close()
    except OSError:
        pass  # Handle the case where the stream is not open

    # Close the wave file if it's open
    if wav_file:
        wav_file.close()

    # Terminate the PyAudio object
    if audio:
        audio.terminate()

    # Stop the asyncio event loop safely
    if event_loop.is_running():
        event_loop.call_soon_threadsafe(event_loop.stop)

    # Close the Tkinter window and exit the application
    root.destroy()

# Keep the Tkinter main loop in the main thread
def main():
    root.mainloop()

# Function to initialize and start the background event loop
def start_async_loop(loop):
    global event_loop
    event_loop = loop  # Store the event loop in a global variable
    asyncio.set_event_loop(loop)
    loop.run_forever()


## GUI
# Create the main window
root = tk.Tk()
root.title("Intelligent Driving Assistance System")
root.configure(bg="#2c2f35")

# Set the protocol for handling the window close event
root.protocol("WM_DELETE_WINDOW", on_closing)

# Get the screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Set the window size to full screen (without being in maximized mode)
root.geometry(f"{screen_width}x{screen_height}")

# Create a frame to hold the buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=50)

# Create the "New Conversation" button
btn_new_conversation = tk.Button(button_frame, text="New\nChat", 
                                 font=("Verdana", 32), 
                                 command=new_session, 
                                 height=8, width=6, bg="#406ce5", fg="black")
btn_new_conversation.pack(side=tk.LEFT)

# Create the "Start listening" button
btn_listening = tk.Button(button_frame, text="Start listening",
                          font=("Verdana", 64), command=toggle_state,
                          height=4, width=20, bg="#3f704d", fg="black")
btn_listening.pack(side=tk.LEFT, padx=20)

# Create a frame to hold the text box and scrollbar
text_frame = tk.Frame(root)
text_frame.pack(pady=20)

# Create a multi-line text box with around 10 lines
text_box = tk.Text(text_frame, font=("Verdana", 16), height=10, width=80, wrap="word")
text_box.pack(side=tk.LEFT)

# Create a vertical scrollbar linked to the text box
scrollbar = tk.Scrollbar(text_frame, command=text_box.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Link the scrollbar to the text box
text_box.config(yscrollcommand=scrollbar.set)


# Run the Tkinter main loop and asyncio event loop
if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    threading.Thread(target=start_async_loop, args=(loop,)).start()
    main()
