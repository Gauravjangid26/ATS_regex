from dotenv import load_dotenv
import streamlit as st
import os
import io
import base64
from PIL import Image
import pdf2image
import google.generativeai as genai
from fpdf import FPDF
import requests
import librosa
import numpy as np
import tempfile
from pydub import AudioSegment
from faster_whisper import WhisperModel
import PyPDF2
import soundfile as sf

import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename='token_usage.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def log_token_usage(feature_name, prompt, response):
    """Log token usage information"""
    try:
        # Estimate tokens (Gemini doesn't provide exact token count in response)
        input_tokens = len(prompt.split()) // 0.75  # Approximate word to token conversion
        output_tokens = len(response.split()) // 0.75 if response else 0
        
        log_message = (
            f"FEATURE: {feature_name} | "
            f"INPUT_TOKENS: ~{int(input_tokens)} | "
            f"OUTPUT_TOKENS: ~{int(output_tokens)} | "
            f"TOTAL_TOKENS: ~{int(input_tokens + output_tokens)}"
        )
        logging.info(log_message)
    except Exception as e:
        logging.error(f"Error logging token usage: {e}")



# Load environment variables
load_dotenv()

# Configure Google Gemini API
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("GOOGLE_API_KEY not found. Please set it in your environment variables.")
    st.stop()

genai.configure(api_key=API_KEY)

# Streamlit Page Config
st.set_page_config(page_title="ATS Resume Analyzer", layout="wide")
st.markdown("""<h1 style='text-align: center;'>📌 AI-Powered Resume ATS & Learning Hub</h1>""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.image("logo.png", width=200)
st.sidebar.title("Navigation")
selected_tab = st.sidebar.radio("Choose a Feature", [
    "🏆 Resume Analysis", "📚 Question Bank", "📊 DSA & Data Science", "🎤 Speech Analysis","🎥Video Analysis","🔝Top 5 MNCs","🗣️ Mock Interview","🛠️ Code Debugger"
])
import tempfile
from faster_whisper import WhisperModel
import PyPDF2

# Function to load resume content from PDF
def load_resume(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        resume_text = "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        return resume_text
    except Exception as e:
        st.error(f"Error reading resume: {e}")
        return None

# Function to extract audio from video
def extract_audio(video_path):
    try:
        temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        audio = AudioSegment.from_file(video_path)  # Automatically detects format
        audio.export(temp_audio_path, format="wav")
        return temp_audio_path
    except Exception as e:
        st.error(f"Error extracting audio: {e}")
        return None

# Function to transcribe audio
def transcribe_audio(audio_path, model_size="small"):
    try:
        model = WhisperModel(model_size)  # Allow dynamic model selection
        segments, _ = model.transcribe(audio_path)
        return " ".join(segment.text for segment in segments)
    except Exception as e:
        st.error(f"Error in transcription: {e}")
        return None

# Function to get AI feedback from Gemini
def get_gemini_feedback(transcribed_text, resume_content):
    prompt = f"""
    Based on the candidate's resume below:
    {resume_content}

    The following is the transcription of their interview response:
    {transcribed_text}

    Please evaluate their response and provide:
    1. Strengths of their answer.
    2. Areas for improvement.
    3. A score out of 10.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([prompt])
        log_token_usage("get gemini feedback", prompt,response)
        return response.text if response else "No response from Gemini."
    except Exception as e:
        st.error(f"Error generating feedback: {e}")
        return None

def ensure_font_exists(font_path):
    """Check if the font file exists; if not, download it."""
    if not os.path.exists(font_path):
        url = "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf"
        st.info(f"Downloading DejaVuSans.ttf from {url} ...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(font_path, "wb") as f:
                f.write(response.content)
            st.success(f"Downloaded DejaVuSans.ttf to {font_path}")
        else:
            raise FileNotFoundError(f"Font file not found at {font_path} and download failed with status code {response.status_code}.")

def get_gemini_response(input_text, pdf_content, prompt):
    """Generate a response using Google Gemini API."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input_text, pdf_content[0], prompt])
    response_text = response.text if response else "No response from Gemini."
    
    # Log token usage
    log_token_usage("get_gemini_response", prompt, response_text)
    return response_text

def get_gemini_response_question(prompt):
    """Generate a response using Google Gemini API."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([prompt])
    response_text = response.text if response else "No response from Gemini."
    
    # Log token usage
    log_token_usage("get_gemini_response_question", prompt, response_text)
    return response_text

def input_pdf_setup(uploaded_file):
    """Convert first page of uploaded PDF to an image and encode as base64."""
    if uploaded_file is not None:
        uploaded_file.seek(0)  # Reset file pointer
        images = pdf2image.convert_from_bytes(uploaded_file.read())
        first_page = images[0]

        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        pdf_parts = [{
            "mime_type": "image/jpeg",
            "data": base64.b64encode(img_byte_arr).decode()  # Encode to base64
        }]
        return pdf_parts
    else:
        raise FileNotFoundError("No File Uploaded")

def generate_pdf(updated_resume_text):
    """Generate a downloadable PDF file with Unicode support."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Define correct font path and ensure it exists
    import os
    font_path = os.path.join(os.getcwd(), "fonts/DejaVuSans.ttf")
    
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.set_font("DejaVu", size=12)

    # Wrap long text in a multi-cell that fits A4 width
    pdf.multi_cell(190, 10, updated_resume_text, align="L")

    pdf_output_path = "updated_resume.pdf"
    pdf.output(pdf_output_path, "F")
    return pdf_output_path


def extract_audio(video_path):
    """Extracts audio from a video file and saves it as a WAV file."""
    audio_path = video_path.replace(".mp4", ".wav")
    try:
        audio = AudioSegment.from_file(video_path)
        audio.export(audio_path, format="wav")
        return audio_path
    except Exception as e:
        st.error(f"❌ Error extracting audio: {e}")
        return None

def analyze_pitch(audio_path):
    """Analyzes pitch and confidence from an audio file."""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        if len(y) == 0:
            st.error("❌ No valid audio detected.")
            return None, None, None

        # Extract pitch using YIN
        f0 = librosa.yin(y, fmin=75, fmax=300)
        valid_f0 = f0[f0 > 0]

        if len(valid_f0) == 0:
            st.error("❌ No valid pitch detected.")
            return None, None, None

        avg_pitch = np.mean(valid_f0)
        pitch_variability = np.std(valid_f0)
        confidence_score = max(0, min(100, 100 - (pitch_variability * 1.2)))

        return avg_pitch, pitch_variability, confidence_score
    except Exception as e:
        st.error(f"❌ Error in pitch analysis: {e}")
        return None, None, None

def get_confidence_label(confidence_score):
    """Determines confidence label based on the confidence score."""
    if confidence_score > 75:
        return "High 🎯", "✅"
    elif confidence_score > 50:
        return "Moderate ⚠️", "🟡"
    else:
        return "Low ❌", "🔴"

# Resume Analysis
if selected_tab == "🏆 Resume Analysis":
    st.subheader("🔍 Resume ATS Analysis")
    input_text = st.text_area("Job Description:", key="input")
    uploaded_file = st.file_uploader("Upload your resume (PDF)...", type=['pdf'])

    if uploaded_file:
        st.success("✅ PDF Uploaded Successfully.")
        pdf_content = input_pdf_setup(uploaded_file)

        # Define buttons
        col0,col1, col2, col3, col4 = st.columns(5)
        with col0:
            submit_tell_me_about=st.button("Tell Me About Resume")
        with col1:
            submit_match = st.button("Percentage Match")
        with col2:
            submit_learning = st.button("Personalized Learning Path")
        with col3:
            submit_update = st.button("Update Resume & Download")
        with col4:
            submit_interview = st.button("Generate Interview Questions")
        input_prompts = {
        "Tell_me_about_resume":"""
        You are an expert resume writer with deep knowledge of Data Science, Full Stack, Web Development, 
        Big Data Engineering, DevOps, and Data Analysis. Your task is to refine and optimize the provided resume 
        according to the job description. Ensure the new resume:
        - Highlights relevant experience and skills.
        - Optimizes for ATS (Applicant Tracking Systems).
        - Uses strong action words and quantifiable achievements.
        - Incorporates key industry keywords.
        """,

        "percentage_match": """
        You are an ATS evaluator. Provide:
        1. An overall match percentage.
        2. Breakdown: 
        - Skills match (% weight)
        - Experience match (% weight)
        - Keyword relevance (% weight)
        3. What can be improved to increase the match?

        """,

        "personalized_learning": """
        You are an experienced learning coach and technical expert. Create a 6-month personalized study plan 
        for an individual aiming to excel in [Job Role], focusing on the skills, topics, and tools specified 
        in the provided job description. Ensure the study plan includes:
        - A list of topics and tools for each month.
        - Suggested resources (books, online courses, documentation).
        - Recommended practical exercises or projects.
        - Periodic assessments or milestones.
        - Tips for real-world applications.
        """,

        "resume_update": """
        You are an expert resume writer with deep knowledge of Data Science, Full Stack, Web Development, 
        Big Data Engineering, DevOps, and Data Analysis. Your task is to refine and optimize the provided resume 
        according to the job description. Ensure the new resume:
        - Highlights relevant experience and skills.
        - Optimizes for ATS (Applicant Tracking Systems).
        - Uses strong action words and quantifiable achievements.
        - Incorporates key industry keywords.
        """,

        "interview_questions": """
        You are an AI-powered interview coach.
        Generate 10 interview questions based on the given job description, 
        focusing on the required skills and expertise.
        """,

        "question_bank": """
        Generate {num_questions} {level}-level interview questions on {topic} with answers.
        """
    }

        if submit_tell_me_about and uploaded_file:
            response=get_gemini_response(input_text,pdf_content,input_prompts["Tell_me_about_resume"])
            st.subheader("Tell_me_about_resume:")
            st.write(response)
            
        elif submit_match and uploaded_file:
            response = get_gemini_response(input_text, pdf_content, input_prompts["percentage_match"])
            st.subheader("Percentage Match:")
            st.write(response)

        elif submit_learning and uploaded_file:
            response = get_gemini_response(input_text, pdf_content, input_prompts["personalized_learning"])
            st.subheader("Personalized Learning Path:")
            st.write(response)

        elif submit_update and uploaded_file:
            response = get_gemini_response(input_text, pdf_content, input_prompts["resume_update"])
            if response:
                pdf_path = generate_pdf(response)
                with open(pdf_path, "rb") as pdf_file:
                    pdf_bytes = pdf_file.read()
                    b64_pdf = base64.b64encode(pdf_bytes).decode()
                    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="Updated_Resume.pdf">Download Updated Resume</a>'
                    st.markdown(href, unsafe_allow_html=True)
            else:
                st.error("Error generating updated resume.")
        elif submit_interview and uploaded_file:
            response=get_gemini_response(input_text, pdf_content, input_prompts["question_bank"])
            st.subheader("Generate Interview Questions")
            st.write(response)

# Question Bank
elif selected_tab == "📚 Question Bank":
    st.subheader("📘 AI-Generated Interview Questions")
    job_role = st.selectbox("🎯 Choose Role", ["Data Scientist", "Data Engineer", "Data Analyst"])
    topic = st.selectbox("📌 Select Topic", ["SQL", "Machine Learning", "Big Data"])
    level = st.radio("🔥 Difficulty Level", ["Easy", "Medium", "Hard"])
    num_questions = st.slider("🎯 Number of Questions", 1, 10, 5)

    if st.button("🔍 Generate Questions") and job_role and topic and level:
        all_questions = ""
        prompt = f"Generate {num_questions} {level}-level interview questions on {topic} with answers."
        response = get_gemini_response_question(prompt)
        all_questions += f"\n\n### {topic} ({level})\n" + response
        st.subheader("Generated Questions:")
        st.write(all_questions)   

# DSA & Data Science Learning
elif selected_tab == "📊 DSA & Data Science":
    st.subheader("🚀 DSA for Data Science")
    dsa_level=st.selectbox("Select Difficulty Level:",["Easy","Medium","Hard"])
    num_questions=10
    prompt_dsa = f"""
    Generate {num_questions} {dsa_level}-level Data Structures and Algorithms (DSA) questions 
    specifically relevant for Data Science roles. Focus on concepts such as:
    - Arrays, Linked Lists, and Hash Maps
    - Searching and Sorting (QuickSort, MergeSort, Binary Search)
    - Dynamic Programming (Knapsack, LCS, etc.)
    - Graph Algorithms (BFS, DFS, Dijkstra's, PageRank)
    - Trees and Tries (Binary Search Trees, Heaps)
    - String Manipulation and Pattern Matching (KMP, Rabin-Karp)
    - Time Complexity and Optimization Techniques

    For each question, provide:
    1. A clear problem statement
    2. Constraints and example test cases
    3. A detailed explanation of the optimal approach
    4. Python code implementation
    """

    if st.button(f"Generate {dsa_level} Questions"):
        response = get_gemini_response_question(prompt_dsa)
        st.subheader(f"{dsa_level}-Level DSA Questions for Data Science")
        st.write(response)

        dsa_topic=st.selectbox("Select DSA Topic:",["Array","Recursion","Linkedlist","Queue","Tree","Graphs","Dynamic Programming"])
        dsa_topic_prompt={"""Teach me {dsa_topic} with case studies.

        Explain {dsa_topic} in detail, covering its concept, importance, and real-world applications. Provide:

        Concept Explanation – Define {dsa_topic} and explain its significance in programming and problem-solving.
        Case Studies – Provide at least two real-world case studies demonstrating the use of {dsa_topic} in domains like Data Science, Machine Learning, Web Development, or System Design.
        Problem-Solving Approach – Explain how {dsa_topic} helps solve practical problems and compare it with alternative methods.
        Code Implementation – Provide a Python implementation, with step-by-step explanations and test cases.
        Best Practices & Optimizations – Discuss common pitfalls, best practices, and performance optimizations.
        Keep the explanation structured, beginner-friendly, and engaging, with clear examples and insights for experienced programmers and generate new question on every cilck with respect to previous."""}

    

    

# Speech Analysis

elif selected_tab == "🎤 Speech Analysis":
    st.subheader("🎙️ Speech & Video Interview Analysis")
    uploaded_audio = st.file_uploader("🎤 Upload an audio/video file", type=['mp3', 'mp4'])
    uploaded_video = st.file_uploader("Upload your video (MP4, AVI, MOV)...", type=['mp4', 'avi', 'mov'])

    if uploaded_video:
        st.success("✅ Video Uploaded Successfully.")
    
        if st.button("Submit"):
            # Save video temporarily
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_video.write(uploaded_video.read())
            temp_video_path = temp_video.name

            # Extract audio
            audio_path = extract_audio(temp_video_path)
            if not audio_path:
                st.stop()

            # Analyze pitch & confidence
            avg_pitch, pitch_variability, confidence_score = analyze_pitch(audio_path)
            if avg_pitch is None:
                st.stop()

            # Determine confidence level
            confidence_label, confidence_color = get_confidence_label(confidence_score)

            # Display Results
            st.write("### 🎙️ Speech Analysis Results")
            st.write(f"**Average Pitch:** {avg_pitch:.2f} Hz")
            st.write(f"**Pitch Variability:** {pitch_variability:.2f}")
            st.write(f"**Confidence Score:** {confidence_score:.1f}/100 {confidence_color} ({confidence_label})")

            # Insights & Recommendations
            st.subheader("📊 Insights & Recommendations")
            if confidence_score > 75:
                st.success("✅ Strong confidence! Keep up the good work.")
            elif confidence_score > 50:
                st.warning("⚠️ Moderate confidence. Try maintaining a steady voice for better clarity.")
            else:
                st.error("❌ Low confidence. Practice speaking with a stable tone and controlled pitch.")



#video analysis
elif selected_tab == "🎥Video Analysis":
    st.subheader("Video And Feedback ")
    st.write("Upload your recorded interview video and resume to receive AI-powered feedback.")

    # Upload resume
    uploaded_resume = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
   
    resume_content = None
    if uploaded_resume:
        resume_content = load_resume(uploaded_resume)
        if resume_content:
            st.success("✅ Resume uploaded successfully!")

    # Upload interview video
    uploaded_video = st.file_uploader("Upload your interview video (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

    if uploaded_video and resume_content:
        st.video(uploaded_video)
        temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.read())
        st.success("✅ Video uploaded successfully!")

        # Extract audio
        audio_path = extract_audio(temp_video_path)
        if audio_path:
            st.success("🎵 Audio extracted successfully!")

            # Allow user to select Whisper model size
            model_size = st.selectbox("Select Whisper Model Size", ["tiny", "base", "small", "medium"], index=2)
            st.write("Model selected:", model_size)

            # Transcribe audio
            with st.spinner("Transcribing audio..."):
                transcribed_text = transcribe_audio(audio_path, model_size)
            
            if transcribed_text:
                st.text_area("📝 Transcribed Text:", transcribed_text, height=150)

                # Get AI feedback
                if st.button("Submit for AI Feedback"):
                    with st.spinner("Generating AI feedback..."):
                        feedback = get_gemini_feedback(transcribed_text, resume_content)
                    st.subheader("AI Feedback")
                    st.write(feedback)



elif selected_tab == "🔝Top 5 MNCs":
    
    selected_company = st.selectbox("Select a company:", ["Amazon", "Google", "Meta", "IBM", "Nvidia"])
    uploaded_file = st.file_uploader("Upload your resume (PDF)...", type=['pdf'])
        
    if uploaded_file and selected_company:
        st.success("✅ PDF Uploaded Successfully.")
        pdf_content = input_pdf_setup(uploaded_file)  # Extract resume content

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            project_btn = st.button("Projects Required", key=f"{selected_company}_projects")

        with col2:
            skills_btn = st.button("Skills Required", key=f"{selected_company}_skills")

        with col3:
            recommend_btn = st.button("Recommendations", key=f"{selected_company}_recommend")

        with col4:
            match_btn = st.button("Resume Match Score", key=f"{selected_company}_match")

        prompt_company = {
            "Projects": f"""
            You are a Data Science career advisor specializing in {selected_company}. 
            Analyze the provided resume and job description. Suggest **real-world projects** 
            that would strengthen the candidate’s profile for a Data Science role at {selected_company}.
            """,
            
            "Skills": f"""
            You are a recruiter at {selected_company}. Based on the provided resume, 
            analyze missing **technical and soft skills** required for a Data Science position at {selected_company}. 
            Highlight skills the candidate already has and suggest improvements.
            """,
            
            "Recommendations": f"""
            You are an expert in hiring Data Scientists at {selected_company}. 
            Based on the resume, provide **personalized recommendations** on:
            - How to tailor the resume to align better with {selected_company}.
            - How to optimize the LinkedIn profile for visibility.
            - Additional resources (books, courses, projects) to improve chances of selection.
            """,

            "MatchScore": f"""
            You are an ATS (Applicant Tracking System) evaluator for {selected_company}.  
            Compare the provided **resume** with a standard **Data Scientist job description at {selected_company}**.  
            Provide a **match score (out of 100)** based on:
            - Relevant experience  
            - Required technical skills (Python, SQL, ML, etc.)  
            - Soft skills (communication, teamwork, etc.)  
            - Projects and past work  

            Also, suggest **specific improvements** to increase the match percentage.
            """
        }
        input_text=""

        if project_btn:
            response = get_gemini_response(input_text, pdf_content, prompt_company["Projects"])
            st.subheader(f"{selected_company} - Projects Required")
            st.write(response)

        elif skills_btn:
            response = get_gemini_response(input_text, pdf_content, prompt_company["Skills"])
            st.subheader(f"{selected_company} - Skills Required")
            st.write(response)

        elif recommend_btn:
            response = get_gemini_response(input_text, pdf_content, prompt_company["Recommendations"])
            st.subheader(f"{selected_company} - Recommendations")
            st.write(response)

        elif match_btn:
            response = get_gemini_response(input_text, pdf_content, prompt_company["MatchScore"])
            st.subheader(f"{selected_company} - Resume Match Score")
            st.write(response)
    else:
        st.warning("Please upload your resume to get personalized insights.")
        


# Mock Interview Button




# Mock Interview Section
if selected_tab == "🗣️ Mock Interview":
    model = genai.GenerativeModel('gemini-1.5-flash')
    data_science_topics = [
        "Machine Learning", "Deep Learning", "Data Analysis",
        "Statistics", "NLP", "Computer Vision", "Big Data", "Feature Engineering"
    ]

    st.subheader("🎤 AI-Powered Mock Interview")
    st.write("Press 'Start Mock' to begin. You will hear a question, respond via voice, and receive AI feedback.")

    # Select Technical Interview Topics
    selected_topics = st.multiselect("Select Data Science Topics for Technical Interview:", data_science_topics)

    # Define General Interview Questions
    general_questions = [
        "Tell me about yourself.",
        "What are your strengths and weaknesses?",
        "Describe a challenging project you've worked on.",
        "Where do you see yourself in five years?",
        "Why should we hire you?"
    ]

    all_scores = []  # Store scores for final evaluation

    # Start Mock Interview
    if st.button("🎙️ Start Mock"):
        for i, question in enumerate(general_questions):
            st.write(f"🔹 **Question {i+1}:** {question}")

            # Upload Audio File
            audio_file = st.file_uploader(f"🎙️ Upload Your Response for Question {i+1}:", type=["wav", "mp3"])

            if audio_file:
                # Convert MP3 to WAV (if necessary) using PyDub
                temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
                
                if audio_file.type == "audio/mpeg":  # MP3 file
                    audio = AudioSegment.from_mp3(audio_file)
                    audio.export(temp_audio_path, format="wav")
                else:  # WAV file
                    with open(temp_audio_path, "wb") as f:
                        f.write(audio_file.read())

                # Transcribe Audio using Whisper
                whisper_model = WhisperModel("small")
                segments, _ = whisper_model.transcribe(temp_audio_path)
                transcribed_text = " ".join(segment.text for segment in segments)

                st.write("📝 **Your Response:**", transcribed_text)

                # Get AI Feedback
                prompt = f"""
                Based on the following interview question:
                **{question}**

                Candidate's Response:
                **{transcribed_text}**

                Please evaluate their response and provide:
                - Strengths of the answer
                - Areas for improvement
                - A score out of 10
                """

                feedback = model.generate_content([prompt])
                feedback_text = feedback.text
                score = int([s for s in feedback_text.split() if s.isdigit()][0])  # Extracting score
                all_scores.append(score)

                st.write("🧠 **AI Feedback:**", feedback_text)
                time.sleep(3)

        # TECHNICAL INTERVIEW SECTION
        if selected_topics:
            st.subheader("🛠️ Technical Interview")

            # AI generates technical questions based on selected topics
            tech_prompt = f"Generate 3 technical interview questions based on these Data Science topics: {', '.join(selected_topics)}."
            tech_questions = model.generate_content([tech_prompt]).text.split("\n")

            for j, tech_question in enumerate(tech_questions):
                if tech_question.strip():
                    st.write(f"💻 **Technical Question {j+1}:** {tech_question}")

                    # Upload response
                    tech_audio_file = st.file_uploader(f"🎙️ Upload Your Response for Technical Question {j+1}:", type=["wav", "mp3"])

                    if tech_audio_file:
                        # Convert MP3 to WAV if necessary
                        temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
                        
                        if tech_audio_file.type == "audio/mpeg":
                            audio = AudioSegment.from_mp3(tech_audio_file)
                            audio.export(temp_audio_path, format="wav")
                        else:
                            with open(temp_audio_path, "wb") as f:
                                f.write(tech_audio_file.read())

                        # Transcribe response
                        segments, _ = whisper_model.transcribe(temp_audio_path)
                        tech_response = " ".join(segment.text for segment in segments)

                        st.write("📝 **Your Response:**", tech_response)

                        # AI feedback on technical response
                        tech_feedback_prompt = f"""
                        Question: **{tech_question}**

                        Candidate's Response:
                        **{tech_response}**

                        Evaluate the response, providing:
                        - Strengths
                        - Areas for improvement
                        - A score out of 10
                        """

                        tech_feedback = model.generate_content([tech_feedback_prompt])
                        tech_feedback_text = tech_feedback.text
                        tech_score = int([s for s in tech_feedback_text.split() if s.isdigit()][0])  # Extracting score
                        all_scores.append(tech_score)

                        st.write("🧠 **AI Feedback:**", tech_feedback_text)
                        time.sleep(3)

        # End Interview Button
        if st.button("🚀 End Interview"):
            avg_score = np.mean(all_scores) if all_scores else 0  # Calculate final performance score
            st.subheader("🏆 Final Interview Performance")
            st.write(f"💡 **Your Overall Score:** {round(avg_score, 1)}/10")
            
            if avg_score >= 8:
                st.success("🎉 Excellent performance! You're well-prepared for real interviews.")
            elif avg_score >= 6:
                st.warning("👍 Good job! Some improvements needed.")
            else:
                st.error("🚀 Keep practicing! Focus on refining your answers.")
    
elif selected_tab == "🛠️ Code Debugger":
    st.header("🛠️ Python Code Debugger")

    user_code = st.text_area("Paste your Python code below:", height=300)

    if st.button("Check & Fix Code"):
        if user_code.strip() == "":
            st.warning("Please enter some code.")
        else:
            with st.spinner("Analyzing and fixing code..."):
                prompt = f"""
                Analyze the following Python code for bugs, syntax errors, and logic errors.
                If it has issues, correct them. Return the fixed code and briefly explain the changes made.

                Code:
                ```python
                {user_code}
                ```
                """

                try:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = model.generate_content([prompt])
                    
                    if response:
                        response_text = response.text  # Get the actual text response
                        log_token_usage("Code Debugger", prompt, response_text)
                        
                        st.subheader("✅ Corrected Code")
                        st.code(response_text, language="python")
                    else:
                        st.error("No response from Gemini.")
                except Exception as e:
                    st.error(f"Error: {e}")
                    logging.error(f"Code Debugger Error: {str(e)}")


# Custom CSS for bottom-right placement and pop-up effect
custom_css = """
<style>
    .bottom-right {
        position: fixed;
        bottom: 10px;
        right: 10px;
        background-color: rgba(0, 0, 0, 0.7);
        color: white;
        padding: 10px 15px;
        border-radius: 10px;
        font-size: 14px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease-in-out;
    }
    
    .bottom-right:hover {
        transform: scale(1.1);
    }
</style>
<div class="bottom-right"> <b>Built by AI Team of Regex Software </b></div>
"""

# Inject CSS into Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)
