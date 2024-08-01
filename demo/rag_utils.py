from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

store = {}          # Manage chat history

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

system_prompt = (
    "You are an intelligent driver assistant called IDAS, your task is to answer the questions that are asked of you" 
    "If the question is about the vehicle, use the provided context obtained from the car manual" 
    "If you don’t know the answer even with the context provided say 'I don't know the answer'"
    "Don’t try to make up an answer."
    "Respond in a concrete way, provide the information extracted and summarized from the context"
    "Do not say that the information appear in the manual for the user to search unless that is the user desire"
    "Keep the answer as concise as possible."
    "\n\n"
    "{context}"
)