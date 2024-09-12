import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain, RetrievalQA
import logging, os, re

# Set OpenAI API key from environment
os.environ["OPENAI_API_KEY"] = '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'

# Initialize the OpenAI API and LangChain components
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_llm = OpenAI(temperature=0.7, max_tokens=256, api_key=openai_api_key)

# Setting up the embeddings and vector store for Retrieval-Aware Generation (RAG)
embeddings = OpenAIEmbeddings()

# Multiple file loader function
def load_multiple_documents(file_paths):
    documents = []
    for file_path in file_paths:
        if file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        elif file_path.endswith('.csv'):
            loader = CSVLoader(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        documents.extend(loader.load())  # Load documents from each file
    return documents

# List of text and CSV files
file_paths = [
    "field_suggestions.txt",
    "University_programs_of_pakistan.txt",
    "heceng_uni.txt",
    "hecgen_uni.txt",
    "hecmedical_uni.txt",
    "hecbusiness_uni.txt",
    "hecarts_uni.txt",
    "affiliatedcollgpakistan.txt",
    "affiliated_collegeslist.txt",
]

# Load all the documents from the files
documents = load_multiple_documents(file_paths)

# Create vector store from the loaded documents
vectorstore = FAISS.from_documents(documents, embeddings)

# Create the retriever using LangChain
retriever = vectorstore.as_retriever()

# Create a PromptTemplate for personality analysis
personality_prompt_template = PromptTemplate(
    input_variables=[
        'name', 'Matric_background', 'matric_percentage', 'Intermediate_background', 'inter_percentage',
        'University_admission_test', 'working_preference', 'previous_experience', 'age', 'Gender',
        'household_income_range', 'fav_subject', 'career_aspirations', 'personality_description', 'near_city',
    ],
    template="""
    Based on the following information, provide a detailed personality breakdown according to the Meyers-Briggs Type Indicator (MBTI) scale, and suggest suitable degree programs.

    Name: {name}
    Matric background: {Matric_background}
    Matric percentage: {matric_percentage}
    Intermediate background: {Intermediate_background}
    Intermediate percentage: {inter_percentage}
    University admission test: {University_admission_test}
    Working preference: {working_preference}
    Previous experience: {previous_experience}
    Age: {age}
    near city: {near_city}
    Gender: {Gender}
    Household Income Range: {household_income_range}
    Favorite subjects: {fav_subject}
    Career aspirations: {career_aspirations}
    Personality description: {personality_description}
    """
)

def format_as_list(text):
    lines = [f"- {line.strip()}" for line in text.strip().split("\n") if line.strip()]
    return "\n".join(lines)

# Static list of medical programs
medical_programs = [
    "MBBS", "BDS", "Doctor of Pharmacy", "Doctor of Physical Therapy",
    "BSc Nursing", "BSc Medical Laboratory Technology", "BSc Radiology",
    "BSc Anesthesia Technology", "BSc Cardiology Technology", "BSc Optometry"
]

# Create a dictionary for medical field suggestions based on MBTI
medical_mbti_suggestions = {
    "ISFJ": ["Nursing", "Pharmacy", "Pediatrics"],
    "ESFJ": ["Public Health", "General Medicine", "Family Medicine"],
    "INFJ": ["Psychiatry", "Clinical Psychology", "Medical Research"],
    "ENFJ": ["Psychiatry", "Pediatrics", "Healthcare Administration"],
    "ISFP": ["Physical Therapy", "Occupational Therapy", "Radiology"],
    "ESFP": ["Emergency Medicine", "General Surgery", "Obstetrics and Gynecology"],
    "INFP": ["Counseling", "Psychiatry", "Mental Health Therapy"],
    "ENFP": ["Psychiatry", "Pediatrics", "General Practice"]
}

def generate_response(form_data):
    try:
        # Personality analysis
        personality_chain = LLMChain(llm=openai_llm, prompt=personality_prompt_template)
        personality_response = personality_chain.run(form_data)

        # Extract MBTI personality type
        personality_type_match = re.search(r'\b([IE][NS][TF][JP])\b', personality_response)
        personality_type = personality_type_match.group(0) if personality_type_match else "Unknown"

        # Get user inputs
        fav_subject = form_data.get('fav_subject', '').lower()
        inter_background = form_data.get('Intermediate_background', '').lower()

        # Handle suggestions for medical field separately using the dictionary and static medical programs list
        if "medical" in fav_subject:
            if personality_type in medical_mbti_suggestions:
                degree_suggestions = f"Based on your MBTI ({personality_type}), suitable medical degrees are: " \
                                     + ", ".join(medical_mbti_suggestions[personality_type]) + ".\n"
                degree_suggestions += "Other general medical programs: " + ", ".join(medical_programs)
            else:
                degree_suggestions = "No specific medical degree suggestions available for your MBTI type."
        else:
            # Query for non-medical fields
            degree_suggestions_query = f"Give suitable degree programs for {fav_subject} with an intermediate background in {inter_background}."
            retrieval_qa = RetrievalQA.from_chain_type(llm=openai_llm, retriever=retriever, chain_type="stuff")
            degree_suggestions = retrieval_qa.run(query=degree_suggestions_query)

        return degree_suggestions, personality_type

    except Exception as e:
        return f"An unexpected error occurred. Error: {e}", None

# Streamlit UI
def main():
    st.title("Degree Program Suggestion System")

    st.header("Enter your details")

    form_data = {
        'name': st.text_input('Name'),
        'Matric_background': st.selectbox('Matric Background', ['Science', 'Arts', 'Commerce']),
        'matric_percentage': st.slider('Matric Percentage', 0, 100),
        'Intermediate_background': st.selectbox('Intermediate Background', ['Pre-Medical', 'Pre-Engineering', 'Commerce', 'Arts']),
        'inter_percentage': st.slider('Intermediate Percentage', 0, 100),
        'University_admission_test': st.text_input('University Admission Test', placeholder="e.g., SAT, ECAT"),
        'working_preference': st.text_input('Working Preference', placeholder="e.g., office, lab"),
        'previous_experience': st.text_input('Previous Experience'),
        'age': st.number_input('Age', min_value=16),
        'near_city': st.text_input('Nearest City'),
        'Gender': st.selectbox('Gender', ['Male', 'Female']),
        'household_income_range': st.selectbox('Household Income Range', ['Below 2 Million PKR', 'Above 2 Million PKR']),
        'fav_subject': st.text_input('Favorite Subject'),
        'career_aspirations': st.text_area('Career Aspirations'),
        'personality_description': st.text_area('Personality Description'),
    }

    if st.button("Get Suggestions"):
        degree_suggestions, personality_type = generate_response(form_data)
        if personality_type:
            st.subheader(f"Personality Type: {personality_type}")
            st.subheader("Degree Suggestions")
            st.text(degree_suggestions)
        else:
            st.error(degree_suggestions)

if __name__ == "__main__":
    main()

  
