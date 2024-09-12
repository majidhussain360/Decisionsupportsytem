from flask import Flask, render_template, request
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain, RetrievalQA
import logging, os, re
app = Flask(__name__)

# Load the API key from environment variables
os.environ["OPENAI_API_KEY"] = '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'

if not os.environ["OPENAI_API_KEY"]:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

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
    # Split text into lines and wrap each line with <li></li>
    lines = [f"<li>{line.strip()}</li>" for line in text.strip().split("\n") if line.strip()]
    return "<ul>" + "".join(lines) + "</ul>"

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
        # Get form inputs
        inter_percentage = float(form_data.get('inter_percentage', 0))  # Convert to float for comparison
        household_income = form_data.get('household_income_range', 0)  # Income comparison
        near_city = form_data.get('near_city', '').lower()  # City input

        # Personality analysis
        personality_chain = LLMChain(
            llm=openai_llm,
            prompt=personality_prompt_template
        )
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
                # Get MBTI-based medical field suggestions
                degree_suggestions = f"Based on your MBTI ({personality_type}), suitable medical degrees are: " \
                                     + ", ".join(medical_mbti_suggestions[personality_type]) + ".\n"
                # Add static medical programs
                degree_suggestions += "Other general medical programs: " + ", ".join(medical_programs)
            else:
                degree_suggestions = "No specific medical degree suggestions available for your MBTI type."
        else:
            # Query for non-medical fields
            degree_suggestions_query = f"Give suitable degree programs for {fav_subject} with an intermediate background in {inter_background}."
            retrieval_qa = RetrievalQA.from_chain_type(llm=openai_llm, retriever=retriever, chain_type="stuff")
            degree_suggestions = retrieval_qa.run(query=degree_suggestions_query)

        # Apply filtering for degree programs based on the subject
        if "engineering" in fav_subject:
            filtered_suggestions = "Engineering programs:\n" + degree_suggestions
        elif "medical" in fav_subject:
            filtered_suggestions = "Medical programs:\n" + degree_suggestions
        elif "computer science" in fav_subject:
            filtered_suggestions = "Computer science programs:\n" + degree_suggestions
        elif "arts" in fav_subject:
            filtered_suggestions = "Arts programs:\n" + degree_suggestions
        elif "business" in fav_subject:
            filtered_suggestions = "Business programs:\n" + degree_suggestions
        else:
            filtered_suggestions = "General programs:\n" + degree_suggestions

        # Format degree suggestions as a list
        degree_suggestions_list = format_as_list(filtered_suggestions)

        # Personality explanation remains unchanged
        personality_explanation = f"""
        <strong>Personality Explanation:</strong>
        Based on the provided information, your personality type is most likely <strong>{personality_type}</strong>.
        """
        personality_explanation_list = format_as_list(personality_explanation)

        # University suggestions for medical and general subjects
        if inter_percentage < 60 and household_income == 'below 2 million PKR per year':
            try:
                affiliated_colleges_text = open("affiliatedcollgpakistan.txt", "r").readlines()
                filtered_colleges = [line for line in affiliated_colleges_text if near_city in line.lower()]
                college_suggestions_list = format_as_list("\n".join(filtered_colleges)) if filtered_colleges else "<strong>No affiliated colleges found in your area.</strong>"
            except Exception as e:
                logging.error(f"Error loading affiliated colleges: {e}")
                college_suggestions_list = "<strong>Affiliated Colleges:</strong> No data available."
        else:
            try:
                if "engineering" in fav_subject:
                    hec_eng_text = open("heceng_uni.txt", "r").readlines()
                    filtered_eng_universities = [line for line in hec_eng_text if near_city in line.lower()]
                    college_suggestions_list = format_as_list("\n".join(filtered_eng_universities)) if filtered_eng_universities else "<strong>No engineering universities found in your area.</strong>"
                elif "medical" in fav_subject:
                    hec_med_text = open("hecmedical_uni.txt", "r").readlines()
                    filtered_med_universities = [line for line in hec_med_text if near_city in line.lower()]
                    college_suggestions_list = format_as_list("\n".join(filtered_med_universities)) if filtered_med_universities else "<strong>No medical universities found in your area.</strong>"
                elif "business" in fav_subject:
                    hec_business_text = open("hecbusiness_uni.txt", "r").readlines()
                    filtered_business_universities = [line for line in hec_business_text if near_city in line.lower()]
                    college_suggestions_list = format_as_list("\n".join(filtered_business_universities)) if filtered_business_universities else "<strong>No business universities found in your area.</strong>"
                else:
                    hec_general_text = open("hecgen_uni.txt", "r").readlines()
                    filtered_general_universities = [line for line in hec_general_text if near_city in line.lower()]
                    college_suggestions_list = format_as_list("\n".join(filtered_general_universities)) if filtered_general_universities else "<strong>No general universities found in your area.</strong>"
            except Exception as e:
                logging.error(f"Error loading universities: {e}")
                college_suggestions_list = "<strong>Universities:</strong> No data available."

        # Pie chart data (unchanged)
        labels, data = [], []
        if "I" in personality_type:
            labels.append("Introversion")
            data.append(75)
        else:
            labels.append("Extraversion")
            data.append(25)

        if "N" in personality_type:
            labels.append("Intuition")
            data.append(65)
        else:
            labels.append("Sensing")
            data.append(35)

        if "T" in personality_type:
            labels.append("Thinking")
            data.append(70)
        else:
            labels.append("Feeling")
            data.append(30)

        if "J" in personality_type:
            labels.append("Judging")
            data.append(60)
        else:
            labels.append("Perceiving")
            data.append(40)

        pie_chart_data = {"labels": labels, "data": data}

        return personality_explanation_list, degree_suggestions_list, college_suggestions_list, pie_chart_data

    except ValueError as e:
        logging.error(f"ValueError: {e}")
        return f"Validation error occurred. Error: {e}", None, None, None
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return f"An unexpected error occurred. Please try again later. Error: {e}", None, None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/submit', methods=['POST'])
def submit():
    form_data = {
        'name': request.form.get('name', ''),
        'Matric_background': request.form.get('Matric_background', ''),
        'matric_percentage': request.form.get('matric_percentage', ''),
        'matric_passing_year': request.form.get('matric_passing_year', ''),
        'Intermediate_background': request.form.get('Intermediate_background', ''),
        'inter_percentage': request.form.get('inter_percentage', ''),
        'inter_year': request.form.get('inter_year', ''),
        'University_admission_test': request.form.get('University_admission_test', ''),
        'working_preference': request.form.get('working_preference', ''),
        'previous_experience': request.form.get('previous_experience', ''),
        'age': request.form.get('age', ''),
         'near_city': request.form.get('near_city', ''),
        'Gender': request.form.get('Gender', ''),
        'residential': request.form.get('residential', ''),
        'diff_ethinicity': request.form.get('diff_ethinicity', ''),
        'Gaurdian_Qualification': request.form.get('Gaurdian_Qualification', ''),
        'guardian_occupation': request.form.get('guardian_occupation', ''),
        'residential_background': request.form.get('residential_background', ''),
        'household_income_range': request.form.get('household_income_range', ''),
        'family_Support': request.form.get('family_Support', ''),
        'hobbies': request.form.get('hobbies', ''),
        'fav_subject': request.form.get('fav_subject', ''),
        'career_aspirations': request.form.get('career_aspirations', ''),
        'armed_forces_interest': request.form.get('armed_forces_interest', ''),
        'related_interest': request.form.get('related_interest', ''),
        'type_of_skills': request.form.get('type_of_skills', ''),
        'acquired_skill': request.form.get('acquired_skill', ''),
        'soft_skills': request.form.get('soft_skills', ''),
        'personality_description': request.form.get('personality_description', '')
    }

    # Generate personality explanation and degree suggestions using the OpenAI API
    personality_explanation, degree_suggestions_output, university_suggestions, pie_chart_data = generate_response(form_data)

     # Check if pie_chart_data exists to avoid NoneType error
    pie_chart_labels = pie_chart_data["labels"] if pie_chart_data else []
    pie_chart_values = pie_chart_data["data"] if pie_chart_data else []

    return render_template('result.html', form_data=form_data,
                           personality_explanation=personality_explanation,
                           degree_suggestions_output=degree_suggestions_output,
                           university_suggestions=university_suggestions,
                           pie_chart_labels=pie_chart_labels,
                           pie_chart_data=pie_chart_values)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
