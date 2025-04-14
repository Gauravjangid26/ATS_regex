import streamlit as st
import whisper
import google.generativeai as genai
import numpy as np
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av

# Configure Gemini AI
genai.configure(api_key="AIzaSyDCg2NMOruCkIPPkfGO3b5mpBM1G09x3v0")  # Replace with actual API key
model = genai.GenerativeModel('gemini-1.5-flash')

# Load Whisper Model
whisper_model = whisper.load_model("small")

# Function to transcribe real-time audio
def transcribe_live_audio(audio_data):
    try:
        result = whisper_model.transcribe(audio_data)
        return result["text"].strip()
    except Exception as e:
        return f"Error in transcription: {str(e)}"

# Streamlit UI
st.title("🎤 AI Mock Interview with Live Speech-to-Text")

# Sidebar
selected_tab = st.sidebar.radio("📌 Select a Tab", ["🏠 Home", "🗣️ Mock Interview"])

# Initialize session state
if "interview_started" not in st.session_state:
    st.session_state.interview_started = False
    st.session_state.current_question = 0
    st.session_state.responses = []
    st.session_state.scores = []
    st.session_state.follow_up = None

# **MOCK INTERVIEW**
if selected_tab == "🗣️ Mock Interview":
    st.subheader("🎤 Real-time AI Interview")

    # **Start Interview**
    if st.button("🎙️ Start Mock Interview"):
        st.session_state.interview_started = True
        st.session_state.current_question = 0
        st.session_state.responses = []
        st.session_state.scores = []
        st.session_state.follow_up = None

    if st.session_state.interview_started:
        general_questions = [
            "Tell me about yourself.",
            "What are your strengths and weaknesses?",
            "Describe a challenging project you've worked on.",
            "Where do you see yourself in five years?",
            "Why should we hire you?"
        ]

        if st.session_state.current_question < len(general_questions):
            question = general_questions[st.session_state.current_question]
            st.write(f"🔹 **Question {st.session_state.current_question + 1}:** {question}")

            # **Real-Time Speech Input**
            webrtc_ctx = webrtc_streamer(
                key="speech-input",
                mode=WebRtcMode.SENDRECV,
                audio_receiver_size=1024,
                video_frame_callback=None,
                async_processing=True
            )

            # Process speech input
            if webrtc_ctx.audio_receiver:
                try:
                    audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
                    for audio_frame in audio_frames:
                        audio_data = audio_frame.to_ndarray()
                        transcribed_text = transcribe_live_audio(audio_data)

                        if transcribed_text:
                            st.session_state.responses.append(transcribed_text)
                            st.write("📝 **Your Response:**", transcribed_text)

                            # AI Feedback & Follow-up
                            ai_prompt = f"""
                            Interview Question: **{question}**

                            Candidate's Response:
                            **{transcribed_text}**

                            1️⃣ Provide strengths of the response.
                            2️⃣ Suggest areas for improvement.
                            3️⃣ Give a score out of 10.
                            4️⃣ Generate a **follow-up question** to dig deeper.
                            """

                            feedback = model.generate_content([ai_prompt])
                            feedback_text = feedback.text
                            score = int([s for s in feedback_text.split() if s.isdigit()][0])  # Extracting score
                            st.session_state.scores.append(score)

                            # Extract follow-up question
                            lines = feedback_text.split("\n")
                            follow_up = lines[-1] if len(lines) > 3 else None
                            st.session_state.follow_up = follow_up

                            st.write("🧠 **AI Feedback:**", feedback_text)
                            time.sleep(2)

                            # Display follow-up question (if available)
                            if follow_up:
                                st.write("🔄 **Follow-up Question:**", follow_up)

                            # Move to the next question
                            st.session_state.current_question += 1
                except Exception as e:
                    st.warning(f"Listening... {e}")

        else:
            st.write("✅ **General Interview Completed! Moving to Technical Questions...**")

            # **TECHNICAL INTERVIEW**
            if st.button("🚀 End Interview"):
                avg_score = np.mean(st.session_state.scores) if st.session_state.scores else 0
                st.subheader("🏆 Final Interview Performance")
                st.write(f"💡 **Your Overall Score:** {round(avg_score, 1)}/10")

                if avg_score >= 8:
                    st.success("🎉 Excellent performance! You're well-prepared for real interviews.")
                elif avg_score >= 6:
                    st.warning("👍 Good job! Some improvements needed.")
                else:
                    st.error("🚀 Keep practicing! Focus on refining your answers.")

                # Reset state after completion
                st.session_state.interview_started = False
                st.session_state.current_question = 0
                st.session_state.responses = []
                st.session_state.scores = []
                st.session_state.follow_up = None
