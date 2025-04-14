# A5 ATS Resume Expert Applicationß

## Overview
The **ATS Resume Expert** is a Streamlit-based web application designed to assist job seekers in optimizing their resumes for Applicant Tracking Systems (ATS). The application leverages Google's Gemini API for AI-powered resume analysis, personalized learning path generation, and interview preparation. It also includes features for audio analysis of interview responses and a question bank for technical interviews.

---

## Features
1. **Resume Analysis**:
   - **Tell Me About Resume**: Provides a detailed analysis of the uploaded resume.
   - **Percentage Match**: Calculates the match percentage between the resume and the job description.
   - **Personalized Learning Path**: Generates a 6-month study plan based on the job description.
   - **Update Resume & Download**: Optimizes the resume and allows users to download the updated version.

2. **Interview Preparation**:
   - **Generate Interview Questions**: Creates interview questions based on the job description.
   - **Question Bank**: Offers a customizable question bank for Data Science, Data Engineering, and Data Analyst roles.
   - **DSA for Data Science**: Generates Data Structures and Algorithms (DSA) questions tailored for Data Science roles.

3. **Top 5 MNCs Analysis**:
   - **Projects Required**: Suggests real-world projects to strengthen the resume for specific companies.
   - **Skills Required**: Identifies missing skills and provides recommendations.
   - **Recommendations**: Offers personalized advice for tailoring the resume and LinkedIn profile.
   - **Resume Match Score**: Provides a match score and improvement suggestions for specific companies.

4. **Audio Analyzer**:
   - **Speech Analysis**: Analyzes pitch and confidence from uploaded interview videos.
   - **AI Interview Analyzer**: Transcribes interview responses and provides AI-powered feedback.

---

## Installation
1. **Prerequisites**:
   - Python 3.8 or higher
   - Streamlit
   - Google Generative AI API key
   - Required Python libraries: `dotenv`, `PIL`, `pdf2image`, `fpdf`, `requests`, `librosa`, `numpy`, `pydub`, `faster_whisper`, `PyPDF2`

2. **Setup**:
   - Clone the repository.
   - Install the required libraries using:
     ```bash
     pip install -r requirements.txt
     ```
   - Create a `.env` file in the root directory and add your Google API key:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```
   - Run the application using:
     ```bash
     streamlit run app.py
     ```



## Usage
1. **Resume Analysis**:
   - Upload your resume in PDF format.
   - Enter the job description in the provided text area.
   - Click on the respective buttons to get resume analysis, match percentage, learning path, or updated resume.

2. **Interview Preparation**:
   - Use the **Generate Interview Questions** button to create questions based on the job description.
   - Access the **Question Bank** to generate technical questions for specific roles and difficulty levels.
   - Use the **DSA for Data Science** feature to generate DSA questions tailored for Data Science roles.

3. **Top 5 MNCs Analysis**:
   - Select a company from the dropdown menu.
   - Click on the respective buttons to get project suggestions, skills analysis, recommendations, or resume match score.

4. **Audio Analyzer**:
   - Upload an interview video in MP4, AVI, or MOV format.
   - Use the **Submit** button to analyze the speech for pitch and confidence.
   - Use the **AI Interview Analyzer** to transcribe the interview response and get AI-powered feedback.

---

## Code Structure
- **Main Application**:
  - The main application is built using Streamlit and is divided into sections for resume analysis, interview preparation, MNC analysis, and audio analysis.
  - The application uses Google's Gemini API for generating responses and feedback.

- **Functions**:
  - **`get_gemini_response`**: Generates responses using the Gemini API.
  - **`input_pdf_setup`**: Converts the first page of the uploaded PDF to an image and encodes it as base64.
  - **`generate_pdf`**: Generates a downloadable PDF with Unicode support.
  - **`extract_audio`**: Extracts audio from a video file.
  - **`analyze_pitch`**: Analyzes pitch and confidence from an audio file.
  - **`transcribe_audio`**: Transcribes audio using the Whisper model.
  - **`get_gemini_feedback`**: Provides AI-powered feedback based on transcribed text and resume content.

---

## Dependencies
- **Streamlit**: For building the web application.
- **Google Generative AI**: For AI-powered resume analysis and feedback.
- **Pillow (PIL)**: For image processing.
- **pdf2image**: For converting PDF pages to images.
- **fpdf**: For generating PDF files.
- **librosa**: For audio analysis.
- **pydub**: For audio file manipulation.
- **faster_whisper**: For audio transcription.
- **PyPDF2**: For reading PDF files.

---

## Troubleshooting
- **API Key Not Found**: Ensure the `.env` file contains the correct Google API key.
- **Font File Missing**: The application will automatically download the required font file if it is not found.
- **Audio Extraction Issues**: Ensure the uploaded video file is in a supported format (MP4, AVI, MOV).

---

## Future Enhancements
- **Integration with LinkedIn**: Automatically fetch profile data for analysis.
- **Multi-language Support**: Extend support for non-English resumes and interviews.
- **Advanced ATS Optimization**: Incorporate more sophisticated ATS optimization techniques.

---

## Conclusion
The **ATS Resume Expert** application is a comprehensive tool for job seekers to optimize their resumes, prepare for interviews, and receive personalized feedback. By leveraging AI-powered analysis, the application helps users improve their chances of success in the job market.

---

## Repository
[GitHub: ATS-chatbot](https://github.com/Gauravjangid26/ATS-chatbot.git)



# ATS_regex
# ATS_regex
# ATS_regex
