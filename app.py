import os
import re
import json
import numpy as np
import chainlit as cl
import openai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from sentence_transformers import CrossEncoder
from datetime import datetime


# Load OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Constants
CUSTOMIZATION_OPTIONS = {
    "style": ["Crew Neck", "V-Neck", "Long Sleeve", "Tank Top"],
    "gender": ["Male", "Female", "Unisex"],
    "color": ["White", "Black", "Blue", "Red", "Green"],
    "size": ["XS", "S", "M", "L", "XL", "XXL"],
    "printing": ["Screen Printing", "Embroidery", "Heat Transfer", "Direct-to-Garment"]
}

ORDER_STEPS = list(CUSTOMIZATION_OPTIONS.keys())
FAQ_PATH = './data/Tee Customizer FAQ.txt'
CHROMA_DB_PATH = "./tmp/chroma_db"

# Global variables
user_sessions = {}
vectorstore = None
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)

# Prompt Template for FAQ
prompt_template = PromptTemplate(
    input_variables=["user_question", "retrieved_answer"],
    template="""
    You are a helpful assistant for a T-shirt customization store.
    Answer ONLY using the provided FAQ. Do NOT guess or make up answers.

    User Question: "{user_question}"

    FAQ Answer: "{retrieved_answer}"

    If the FAQ does not contain an answer, respond with: "I'm not sure, but you can contact our support team."
    """
)

# Create the LLM Chain
faq_chain = LLMChain(llm=llm, prompt=prompt_template)

def split_faq():
    """Split FAQ text into structured Q&A pairs."""
    with open(FAQ_PATH, "r") as f:
        faq_text = f.read()
    qa_pairs = re.findall(r"(Q:.*?)(?=Q:|$)", faq_text, re.DOTALL)
    structured_faq = []
    for qa in qa_pairs:
        parts = qa.split("A:", 1)
        if len(parts) == 2:
            question = parts[0].strip().replace("Q:", "").strip()
            answer = parts[1].strip()
            structured_faq.append({"question": question, "answer": answer})
    return structured_faq

def initialize_faq_database():
    """Initialize ChromaDB with FAQ questions"""
    global vectorstore
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    if os.path.exists(CHROMA_DB_PATH):
        vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_model)
    else:
        structured_faq = split_faq()
        documents = [Document(page_content=entry["answer"], metadata={"question": entry["question"]}) for entry in structured_faq]
        vectorstore = Chroma.from_documents(documents, embedding_model, persist_directory=CHROMA_DB_PATH)

@cl.on_chat_start
async def start_chat():
    """Initialize FAQ database and welcome the user"""
    global vectorstore
    initialize_faq_database()
    user_sessions[cl.user_session] = {"step": 0, "choices": {}, "order_confirmed": False, "in_support": False}
    await cl.Message(content="Hello! How can I help you today?\n\n Order a custom T-shirt\n Ask a question \n Log a support request").send()

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages from the user"""
    user_input = message.content.strip()
    user_id = cl.user_session

    if user_id not in user_sessions:
        user_sessions[user_id] = {"choices": {}, "order_confirmed": False, "in_support": False, "support_issue": None}

    session = user_sessions[user_id]
    order_state = session["choices"]

    if is_valid_order_choice(user_input, order_state):
        await handle_order_choice(user_input, session, order_state)
        return

    if session["order_confirmed"]:
        await handle_order_confirmation(user_input, session)
        return

    if session.get("in_support", False):
        await handle_support_request(user_input, session)
        return

    if is_support_request(user_input):
        await initiate_support_request(session)
        return

    response = get_llm_response(user_input, order_state)
    await cl.Message(content=response).send()

def is_order_complete(order_state):
    """Check if all order details are filled"""
    return all(step in order_state for step in ORDER_STEPS)

async def confirm_order(session):
    """Confirm the order with the user"""
    if not session["order_confirmed"]:
        session["order_confirmed"] = True
        user_sessions[cl.user_session] = session
        order_summary = "\n".join([f"- {k}: {v}" for k, v in session["choices"].items()])
        await cl.Message(content=f"Your order is complete!\n{order_summary}\n\nWould you like to confirm or cancel?").send()


async def handle_order_confirmation(user_input, session):
    """Handle the user's response to the order confirmation"""
    confirmation_result = get_llm_response(user_input, confirmation_check=True)

    if confirmation_result == "confirm":
        session["choices"] = {}
        session["order_confirmed"] = False
        user_sessions[cl.user_session] = session
        await cl.Message(content="Your order has been placed! How else can I assist you?").send()
    elif confirmation_result == "cancel":
        session["choices"] = {}
        session["order_confirmed"] = False
        user_sessions[cl.user_session] = session
        await cl.Message(content="Your order has been canceled. Let me know if you need anything else!").send()
    else:
        await cl.Message(content="I didn't quite understand. Do you want to confirm or cancel your order? (Yes/No)").send()

def is_support_request(user_input):
    """Check if the user input indicates a support request"""
    return any(word in user_input.lower() for word in ["help", "support", "problem", "issue"])

async def initiate_support_request(session):
    """Initiate a support request"""
    session["in_support"] = True
    user_sessions[cl.user_session] = session
    await cl.Message(content="I'm here to help! Please describe the issue you're facing.").send()

async def handle_support_request(user_input, session):
    """Handle a support request from the user"""
    if session.get("support_issue") is None:
        session["support_issue"] = user_input
        user_sessions[cl.user_session] = session
        await cl.Message(content="Got it! Can you please provide your order number (if applicable)?").send()
    else:
        save_support_request(cl.user_session, session["support_issue"], user_input)
        await cl.Message(content="Your support request has been submitted! Our team will get back to you shortly.").send()
        session["in_support"] = False
        session["support_issue"] = None
        user_sessions[cl.user_session] = session

def get_llm_response(user_query, order_state=None, confirmation_check=False):
    """Generate a response from the LLM based on the user query"""
    if confirmation_check:
        return classify_confirmation(user_query)

    # Check if the user input is a valid order choice
    if is_valid_order_choice(user_query, order_state or {}):
        return handle_order_choice(user_query, order_state)

    # If not a valid order choice, proceed with FAQ lookup
    retrieved_answer, score = retrieve_faq_answer_with_reranking(user_query)
    if retrieved_answer and score >= 0.9:
        return faq_chain.run(user_question=user_query, retrieved_answer=retrieved_answer).strip()

    # If no valid FAQ answer, proceed with the default prompt
    order_summary = build_order_summary(order_state)
    missing_steps = [step for step in ORDER_STEPS if step not in order_state] if order_state else []

    prompt = build_prompt(user_query, order_summary, missing_steps)
    response = llm.invoke(prompt)
    return response.content.strip()

def is_valid_order_choice(user_input, order_state):
    """Check if the user input is a valid order choice"""
    for category, options in CUSTOMIZATION_OPTIONS.items():
        if category not in order_state and user_input.lower() in list(map(str.lower, options)):
            return True

    return False

async def handle_order_choice(user_input, session, order_state):
    """Handle a valid order choice from the user"""
    for category, options in CUSTOMIZATION_OPTIONS.items():
        if user_input.lower() in list(map(str.lower, options)) and category not in order_state:
            order_state[category] = user_input.capitalize()
            session["choices"] = order_state
            user_sessions[cl.user_session] = session

            # Get the next step
            next_step = get_next_order_step(order_state)
            if next_step:
                response = f"It looks like you've selected {user_input.capitalize()} for {category}. The next step is to choose a {next_step}. Here are the available options:\n{', '.join(CUSTOMIZATION_OPTIONS[next_step])}"
            else:
                # If all steps are complete, trigger the confirmation flow
                session["order_confirmed"] = True  # Mark the order as ready for confirmation
                order_summary = "\n".join([f"- {k}: {v}" for k, v in order_state.items()])
                response = f"Your order is complete!\n{order_summary}\nWould you like to confirm or cancel?"

            await cl.Message(content=response).send()
            return
    await cl.Message(content="Invalid option selected.").send()

def get_next_order_step(order_state):
    """Get the next step in the order process"""
    for step in ORDER_STEPS:
        if step not in order_state:
            return step
    return None

def classify_confirmation(user_query):
    """Classify the user's response to the order confirmation"""
    confirmation_prompt = f"""
        You are an intelligent assistant handling a custom T-shirt order.

        The user was asked to confirm or cancel their order. They responded with:
        "{user_query}"

        Classify the user's response as one of the following:
        - "confirm" (if they want to confirm the order)
        - "cancel" (if they want to cancel the order)
        - "uncertain" (if the response is unclear)

        Return only the classification word ("confirm", "cancel", or "uncertain") and nothing else.
        """
    response = llm.invoke(confirmation_prompt).content.strip().lower()
    return response

def build_order_summary(order_state):
    """Build a summary of the current order state"""
    if order_state:
        return "Here is what the user has selected so far:\n" + "\n".join([f"- {k}: {v}" for k, v in order_state.items()])
    return "No selections made yet."

def build_prompt(user_query, order_summary, missing_steps):
    """Build a prompt for the LLM based on the current context"""
    return f"""
    You are a chatbot for an online custom T-shirt store called TeeCustomizer.
    Your job is to:
    - Guide users through ordering a custom T-shirt.
    - Answer FAQs based on store policies.
    - Handle support requests and capture details.

    **T-Shirt Customization Options:**
    {CUSTOMIZATION_OPTIONS}
    **Current Order Progress:**
    {order_summary}

    **Guidelines:**
    - If some details are missing, ONLY ask for the next missing detail: {", ".join(missing_steps) if missing_steps else "None"}.
    - When asking for the missing detail, list all the available options.
    - DO NOT repeat steps that the user has already completed.
    - If the user asks a question unrelated to ordering, answer based on the FAQ database.
    - If the user needs support or has problems with their order, ask for details before submitting a request.

    **User Input (Order Selection or Question):** "{user_query}"
    **Chatbot Response:**
    """

def save_support_request(user_id, issue_summary, order_number=None):
    """Store user support requests in a JSON file"""
    support_request = {
        "user_id": str(user_id),
        "timestamp": datetime.now().isoformat(),
        "issue_summary": issue_summary,
        "order_number": order_number if order_number else "N/A",
    }
    with open("support_requests.json", "a") as file:
        file.write(json.dumps(support_request) + "\n")

def retrieve_faq_answer_with_reranking(query, threshold=0.7):
    """Retrieve and rerank FAQ answers based on the user query"""
    results = vectorstore.similarity_search_with_score(query, k=5)
    if not results:
        return None, 0

    retrieved_answers = [res[0].page_content for res in results]
    best_answer, score = rerank_results(query, retrieved_answers)

    if score < threshold:
        return None, score

    return best_answer, score

def softmax(x):
    """Compute softmax values for an array of logits"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def rerank_results(query, retrieved_answers):
    """Re-rank retrieved answers and normalize scores using softmax"""
    if not retrieved_answers:
        return None, 0

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [[query, answer] for answer in retrieved_answers]
    scores = reranker.predict(pairs)
    softmax_scores = softmax(scores)
    best_idx = np.argmax(softmax_scores)
    return retrieved_answers[best_idx], softmax_scores[best_idx]