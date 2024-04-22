import streamlit as st
from streamlit import session_state
import json
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from PIL import Image
from PyPDF2 import PdfReader
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

session_state = st.session_state
if "user_index" not in st.session_state:
    st.session_state["user_index"] = 0

load_dotenv()

safety_settings = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_json_text():
    text = ""
    for json_file in os.listdir("medical-question-answer-data"):
        if json_file.endswith(".json"):
            with open("medical-question-answer-data//" + json_file) as file:
                datas = json.load(file)
                for data in datas:
                    question = data["question"]
                    answer = data["answer"]
                    text += "Patient: " + question + "\n"
                    text += "Doctor: " + answer + "\n"
    return str(text.encode("utf-8"))


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
  You are a health specialist. Your task is to diagnose the patient's condition based on the provided context and answer the patient's questions. Provide detailed answers to the patient's questions, and if the answer is not in the provided context, generate a response based on the context and the question. Act as a health specialist and offer the best possible advice based on the patient's condition. Provide additional information or advice, ask follow-up questions to gather more details, and do not refer the patient to another doctor, professional, or expert. Handle the patient's queries and concerns professionally, providing information about the patient's condition, precautions, treatments needed, any mental disorders or health issues, and medications or treatments required.
  Act like a health specialist and always provide the advice. Never reveal that you are an AI assistant. If the patient asks for a treatment plan, provide a treatment plan based on the patient's condition and the information provided. If the patient asks for a prognosis, provide a prognosis based on the patient's condition and the information provided. If the patient asks for a second opinion, provide a second opinion based on the patient's condition and the information provided. If the patient asks for precautions, suggest precautions based on the patient's condition and the information provided. If the patient asks for a diagnosis, provide a diagnosis based on the patient's condition and the information provided. If the patient asks for a prescription, provide a prescription based on the patient's condition and the information provided. 
Context:
  {context}?

Question:
  {question}

Answer:

    """

    model = ChatGoogleGenerativeAI(
        model="gemini-pro", temperature=0.4, safety_settings=safety_settings
    )

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def health_specialist(
    question, previous_question=None, previous_response=None, json_file_path="data.json"
):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    if previous_response is None:
        docs = ""
    else:
        docs = new_db.similarity_search(question)
    if previous_question is not None and previous_response is not None:
        additional_context = (
            "Doctor: " + previous_question + "\nPatient: " + previous_response + "\n"
        )
        question = additional_context + "Patient: " + question
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": question}, return_only_outputs=True
    )
    return response["output_text"]


def generate_medical_report(name, previous_questions=None, previous_responses=None):
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-pro")
        prompt = f"Patient: {name}\n\n"
        if previous_questions and previous_responses:
            for question, response in zip(previous_questions, previous_responses):
                prompt += f"Doctor: {question}\nPatient: {response}\n\n"

        else:
            prompt += "Assistant: Can you provide any relevant information about your condition?\nPatient:"

        # Add a request for medical report generation
        prompt += "\n\nGenerate a detailed medical report including any mental disorders and precautions needed."
        response = model.generate_content(prompt, safety_settings=safety_settings)
        return response.text

    except Exception as e:
        st.error(f"Error getting marks: {e}")
        return None


def signup(json_file_path="data.json"):
    st.title("Signup Page")
    with st.form("signup_form"):
        st.write("Fill in the details below to create an account:")
        name = st.text_input("Name:")
        email = st.text_input("Email:")
        age = st.number_input("Age:", min_value=0, max_value=120)
        sex = st.radio("Sex:", ("Male", "Female", "Other"))
        password = st.text_input("Password:", type="password")
        confirm_password = st.text_input("Confirm Password:", type="password")

        if st.form_submit_button("Signup"):
            if password == confirm_password:
                user = create_account(
                    name,
                    email,
                    age,
                    sex,
                    password,
                    json_file_path,
                )
                session_state["logged_in"] = True
                session_state["user_info"] = user
            else:
                st.error("Passwords do not match. Please try again.")


def check_login(username, password, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        for user in data["users"]:
            if user["email"] == username and user["password"] == password:
                session_state["logged_in"] = True
                session_state["user_info"] = user
                st.success("Login successful!")
                return user
        return None
    except Exception as e:
        st.error(f"Error checking login: {e}")
        return None


def initialize_database(json_file_path="data.json"):
    try:
        if not os.path.exists(json_file_path):
            data = {"users": []}
            with open(json_file_path, "w") as json_file:
                json.dump(data, json_file)
    except Exception as e:
        print(f"Error initializing database: {e}")


def create_account(
    name,
    email,
    age,
    sex,
    password,
    json_file_path="data.json",
):
    try:
        # Check if the JSON file exists or is empty
        if not os.path.exists(json_file_path) or os.stat(json_file_path).st_size == 0:
            data = {"users": []}
        else:
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)

        # Append new user data to the JSON structure
        user_info = {
            "name": name,
            "email": email,
            "age": age,
            "sex": sex,
            "password": password,
            "report": None,
            "questions": None,
        }
        data["users"].append(user_info)

        # Save the updated data to JSON
        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        st.success("Account created successfully! You can now login.")
        return user_info
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        st.error(f"Error creating account: {e}")
        return None


def login(json_file_path="data.json"):
    st.title("Login Page")
    username = st.text_input("Username:")
    password = st.text_input("Password:", type="password")

    login_button = st.button("Login")

    if login_button:
        user = check_login(username, password, json_file_path)
        if user is not None:
            session_state["logged_in"] = True
            session_state["user_info"] = user
        else:
            st.error("Invalid credentials. Please try again.")


def get_user_info(email, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
            for user in data["users"]:
                if user["email"] == email:
                    return user
        return None
    except Exception as e:
        st.error(f"Error getting user information: {e}")
        return None


def render_dashboard(user_info, json_file_path="data.json"):
    try:
        # add profile picture
        st.title(f"Welcome to the Dashboard, {user_info['name']}!")

        # Define columns to arrange content
        col1, col2 = st.columns([2, 3])

        with col1:
            st.subheader("Profile Picture:")
            if user_info["sex"] == "Male":
                st.image(
                    "https://www.shutterstock.com/image-vector/male-avatar-profile-picture-use-600nw-193292036.jpg",
                    width=200,
                )
            else:
                st.image(
                    "https://www.shutterstock.com/image-vector/female-profile-avatar-icon-white-260nw-193292228.jpg",
                    width=200,
                )

        with col2:
            st.subheader("User Information:")
            st.write(f"Name: {user_info['name']}")
            st.write(f"Sex: {user_info['sex']}")
            st.write(f"Age: {user_info['age']}")

        # Check if medical report exists
        if user_info["report"] is not None:
            st.markdown("## Medical Report:")
            st.write(f"Report: {user_info['report']}")
        else:
            st.warning("You do not have a medical report yet.")
    except Exception as e:
        st.error(f"Error rendering dashboard: {e}")


def main(json_file_path="data.json"):

    st.sidebar.title("SERENITY - Your Personal Mental Health expert")
    page = st.sidebar.radio(
        "Go to",
        (
            "Signup/Login",
            "Dashboard",
            "Doctor Consultation",
            "View Medical Report",
        ),
        key="pages",
    )

    if page == "Signup/Login":
        st.title("Signup/Login Page")
        login_or_signup = st.radio(
            "Select an option", ("Login", "Signup"), key="login_signup"
        )
        if login_or_signup == "Login":
            login(json_file_path)
        else:
            signup(json_file_path)

    elif page == "Dashboard":
        if session_state.get("logged_in"):
            render_dashboard(session_state["user_info"])
        else:
            st.warning("Please login/signup to view the dashboard.")

    elif page == "Doctor Consultation":
        if session_state.get("logged_in"):
            user_info = session_state["user_info"]
            st.title("Your Personal Mental Health Specialist")
            st.write("Chat with the health specialist to get medical advice.")
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)
                user_index = next(
                    (
                        i
                        for i, user in enumerate(data["users"])
                        if user["email"] == session_state["user_info"]["email"]
                    ),
                    None,
                )
                if user_index is not None:
                    user_info = data["users"][user_index]

            if "messages" not in st.session_state:
                st.session_state.messages = []

            if user_info["questions"] is None:
                previous_response = None
                previous_question = None
            else:
                previous_response = user_info["questions"][-1]["response"]
                previous_question = user_info["questions"][-1]["question"]
            if user_info["questions"] is not None and len(user_info["questions"]) > 0:
                for questions in user_info["questions"]:
                    st.chat_message("Doctor", avatar="🤖").write(questions["question"])
                    st.chat_message("Patient", avatar="👩‍🎨").write(questions["response"])

            if question := st.chat_input("Enter your question here", key="question"):
                with st.chat_message("Patient", avatar="👩‍🎨"):
                    st.markdown(question)

                response = health_specialist(
                    question,
                    previous_question,
                    previous_response,
                )
                with st.chat_message("Doctor", avatar="🤖"):
                    st.markdown(response)

                with open(json_file_path, "r+") as json_file:
                    data = json.load(json_file)
                    user_index = next(
                        (
                            i
                            for i, user in enumerate(data["users"])
                            if user["email"] == session_state["user_info"]["email"]
                        ),
                        None,
                    )
                    if user_index is not None:
                        user_info = data["users"][user_index]
                        if user_info["questions"] is None:
                            user_info["questions"] = []
                        user_info["questions"].append(
                            {"question": question, "response": response}
                        )
                        session_state["user_info"] = user_info
                        json_file.seek(0)
                        json.dump(data, json_file, indent=4)
                        json_file.truncate()
                    else:
                        st.error("User not found.")
                response = None
                st.rerun()

            if st.button("Complete the consultation and generate report"):
                report = generate_medical_report(
                    user_info["name"],
                    [q["question"] for q in user_info["questions"]],
                    [q["response"] for q in user_info["questions"]],
                )
                with open(json_file_path, "r+") as json_file:
                    data = json.load(json_file)
                    user_index = next(
                        (
                            i
                            for i, user in enumerate(data["users"])
                            if user["email"] == session_state["user_info"]["email"]
                        ),
                        None,
                    )
                    if user_index is not None:
                        user_info = data["users"][user_index]
                        user_info["report"] = report
                        session_state["user_info"] = user_info
                        json_file.seek(0)
                        json.dump(data, json_file, indent=4)
                        json_file.truncate()
                    else:
                        st.error("User not found.")
                st.success("Report generated successfully!")
                return

            if st.button("Clear chat"):
                session_state.pop("messages", None)
        else:
            st.warning("Please login/signup to chat.")

    elif page == "View Medical Report":
        if session_state.get("logged_in"):
            st.title("View Medical Report")
            user_info = session_state["user_info"]
            if user_info["report"] is not None:
                st.write(f"Report: {user_info['report']}")
            else:
                st.warning("You do not have a medical report yet.")
        else:
            st.warning("Please login/signup to meditate.")
    else:
        st.error("Invalid page selection.")


if __name__ == "__main__":
    initialize_database()
    main()
