from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import random
import joblib
import numpy as np

load_dotenv()

model = joblib.load("model_ingestion/disease_model.pkl")
le = joblib.load("model_ingestion/label_encoder.pkl")
symptoms_list = joblib.load("model_ingestion/symptoms.pkl")

pending_symptom = False
pending_booking = False
pending_prediction_confirmation = False
pending_query = ""

chat_history = []
def format_history(history):
    formatted = ""
    for role, msg in history[-6:]:   # last 6 messages
        formatted += f"{role}: {msg}\n"
    return formatted

mode = "NORMAL" 

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "Database_db",
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)

llm = init_chat_model("groq:llama-3.3-70b-versatile")

def is_greeting(q):
    return q.lower().strip() in ["hi", "hello", "hey", "hii"]

def is_noise(q):
    return q.lower().strip() in ["sun", "moon", "sky", "ok", "thanks"]

def is_opd_query(q):
    return any(k in q.lower() for k in ["opd", "doctor", "appointment", "book", "consult"])

def is_emergency_query(q):
    return any(k in q.lower() for k in ["chest pain", "breathing", "unconscious", "emergency", "severe"])

def validate_day(docs, doctor_name, day):
    context = "\n".join(d.page_content for d in docs).lower()
    context = context.replace("–", "-")

    doctor_name = doctor_name.lower()
    doctor_name = doctor_name.replace("dr.", "")
    doctor_name = doctor_name.replace("dr", "")
    doctor_name = doctor_name.replace(".", "")
    doctor_name = doctor_name.strip()

    doctor_name = " ".join(doctor_name.split())

    day = day.lower().strip()

    day_map = {
        "monday": "mon",
        "tuesday": "tue",
        "wednesday": "wed",
        "thursday": "thu",
        "friday": "fri",
        "saturday": "sat",
        "sunday": "sun"
    }

    day_variants = [day, day_map.get(day, "")]

    doctor_found = False

    for d in docs:
        doc_text = d.page_content.lower()
        doc_text = doc_text.replace("dr.", "").replace("dr", "").replace(".", "")
        doc_text = " ".join(doc_text.split())

        if doctor_name in doc_text:
            doctor_found = True
            break

    if not doctor_found:
        return False

    for d in day_variants:
        if d and d in context:
            return True

    return False
def is_symptom_query(query):
    prompt = f"""
Classify:
SYMPTOM or OTHER

Message: {query}
"""
    return llm.invoke(prompt).content.strip().upper() == "SYMPTOM"

def extract_symptoms(user_text):
    prompt = f"""
Extract medical symptoms from the sentence below.

Rules:
- Return ONLY symptoms
- Use simple words (fever, cough, headache, etc.)
- No explanations
- Output as comma-separated list

Sentence: {user_text}
"""
    response = llm.invoke(prompt).content
    return [s.strip().lower().replace(" ", "_") for s in response.split(",")]

def create_input_vector(user_symptoms):
    vec = [0] * len(symptoms_list)
    for s in user_symptoms:
        col = f"symptom_{s}"
        if col in symptoms_list:
            vec[symptoms_list.index(col)] = 1
    return np.array(vec).reshape(1, -1)

def predict_disease(user_symptoms):
    vec = create_input_vector(user_symptoms)
    probs = model.predict_proba(vec)[0]
    top = probs.argsort()[-3:][::-1]

    results = []
    for i in top:
        disease = le.inverse_transform([i])[0]
        confidence = round(probs[i] * 100, 2)
        results.append((disease, confidence))
    return results

print("Hospital AI Assistant Started")
print("If you are not sure about what you are experiencing you can type 'predict' so we can make assumptions for you so that u can meet correct doctor.")
print("Press 0 to exit")

while True:
    query = input("You: ")

    if query == "0":
        break

    chat_history.append(("User", query))

    if is_noise(query):
        print("Bot: Please ask medical or hospital-related questions.")
        continue

    if "predict" in query.lower():
        mode = "SYMPTOM"
        print("Bot: Please describe your symptoms.")
        continue

    if mode == "SYMPTOM":
        symptoms = extract_symptoms(query)

        valid = [s for s in symptoms if f"symptom_{s}" in symptoms_list]

        if not valid:
            print("Bot: Please describe clear symptoms like fever, headache.")
            continue

        print("\nBot: Analyzing symptoms...\n")

        results = predict_disease(valid)

        for d, c in results:
            print(f"- {d} ({c}%)")

        top = results[0][0]

        print(f"\nBot: Likely condition: {top}")

        query = f"{top} doctor opd"
        mode = "NORMAL"
        continue

    if "book" in query.lower():
        mode = "BOOKING"
        print("""
Bot: You can book an appointment via:
1. Hospital reception visit
2. Calling: +91-9876543210
3. Email: info@abchospital.com

Do you want me to book an appointment for you? (yes/no)
""")
        continue

    if mode == "BOOKING":
        if query.lower() == "yes":
            doc_name = input("Enter doctor name: Dr. ")
            day = input("Enter day: ")
            timing = input("Enter time: ")

            docs = retriever.invoke(doc_name)

            if not validate_day(docs, doc_name, day):
                print("Bot: Invalid doctor or day.")
                mode = "NORMAL"
                continue

            booking_id = f"APPT-{random.randint(10000, 99999)}"

            print(f"""
                Bot: Appointment Confirmed
                
                Doctor: {doc_name}
                Day: {day}
                
                Request: {pending_query}
                
                Booking Code: {booking_id}
                """)

        else:
            print("Bot: Booking cancelled.")

        mode = "NORMAL"
        continue

    if is_greeting(query):
        print("Bot: Hello! How can I help you today?")
        continue

    if is_opd_query(query) and not is_emergency_query(query):
        query += " opd doctors only"

    if is_emergency_query(query):
        query += " emergency department only"

    docs = retriever.invoke(query)
    docs = [d for d in docs if len(d.page_content.strip()) > 30]

    if not docs:
        print("Bot: I don't have this information.")
        continue

    context = "\n".join(d.page_content for d in docs)

    history_text = format_history(chat_history)

    prompt = f"""
You are a helpful and professional hospital operations assistant.

GOAL:
Provide clear, friendly, and useful responses related to hospitals, healthcare, doctors, appointments, and medical services.

BEHAVIOR RULES:

1. Tone:
- Be polite, friendly, and conversational (not robotic).
- Responses should feel natural, like a helpful hospital assistant.
- Avoid being too short or too long — give medium-length, informative answers.

2. Scope:
- You can answer:
  • Hospital operations (OPD, IPD, emergency, departments)
  • Doctors and appointments
  • Symptoms and general medical guidance (non-diagnostic)
  • Basic healthcare explanations

3. General Knowledge (IMPORTANT):
- If the question is about general hospital concepts (e.g., "What is OPD?"):
  → You ARE allowed to answer using general knowledge
  → Even if it is NOT present in the database

4. Database Usage:
- If the query is about specific hospital data (doctors, timings, availability):
  → Use ONLY the provided context
  → DO NOT invent or guess missing data

5. Missing Data:
- If specific hospital info is not found in context:
  → Say: "This information is not available in hospital records."

6. Irrelevant Queries:
- If the question is completely unrelated (weather, jokes, etc.):
  → Respond politely:
    "I can assist with hospital and healthcare-related queries."

7. Emergency Handling:
- Only suggest emergency care if clearly needed (chest pain, unconscious, severe symptoms)
- Do NOT over-direct users to emergency

8. Response Quality:
- Avoid one-line answers unless question is very simple
- Give structured, easy-to-understand explanations when needed

---

Conversation:
{history_text}

Hospital Context:
{context}

User Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    print("Bot:", response.content)

    chat_history.append(("Bot", response.content))
