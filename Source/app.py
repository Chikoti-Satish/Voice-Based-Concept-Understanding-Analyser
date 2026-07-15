import streamlit as st
import whisper
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from google import genai
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

st.set_page_config(page_title="VBCUA System")
st.title("🗣️ Voice-Based Concept Understanding Analyser")
st.write("Welcome to the VBCUA system! Upload an audio explanation below.")

# 1. Load AI Models (Cached for speed)
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

@st.cache_resource
def load_sbert_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_whisper_model()
sbert_model = load_sbert_model()

# Helper function for fluency analysis
def count_filler_words(text):
    fillers = [" um ", " uh ", " like ", " you know ", " basically ", " actually "]
    count = 0
    text_lower = " " + text.lower() + " "
    for f in fillers:
        count += text_lower.count(f)
    return count

# Helper function to generate PDF using ReportLab
def create_pdf_report(reference, transcription, score, fillers, energy, ai_feedback):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    
    styles = getSampleStyleSheet()
    
    # Custom Styles for Clean Presentation
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=20, spaceAfter=15)
    section_style = ParagraphStyle('SectionStyle', parent=styles['Heading2'], fontSize=14, spaceBefore=12, spaceAfter=6)
    body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'], fontSize=11, leading=16, spaceAfter=8)
    
    # Document Header
    story.append(Paragraph("Voice-Based Concept Understanding Analyser (VBCUA) Report", title_style))
    story.append(Spacer(1, 10))
    
    # Section 1: Concept Metadata
    story.append(Paragraph("1. Concept Reference Mapping", section_style))
    story.append(Paragraph(f"<b>Expected Definition:</b> {reference}", body_style))
    story.append(Spacer(1, 5))
    story.append(Paragraph(f"<b>Student Transcription:</b> {transcription}", body_style))
    
    # Section 2: Metrics Dashboard
    story.append(Paragraph("2. Quantitative Metrics", section_style))
    story.append(Paragraph(f"• <b>Conceptual Similarity Score:</b> {score}%", body_style))
    story.append(Paragraph(f"• <b>Filler Words Counter:</b> {fillers} occurrences", body_style))
    story.append(Paragraph(f"• <b>Average Audio Amplitude/Energy:</b> {energy:.4f}", body_style))
    
    # Section 3: AI Insights
    story.append(Paragraph("3. Intelligent Qualitative Feedback", section_style))
    # Formatting newline blocks from AI response nicely for ReportLab
    formatted_feedback = ai_feedback.replace("\n", "<br/>")
    story.append(Paragraph(formatted_feedback, body_style))
    
    # Build document structure
    doc.build(story)
    buffer.seek(0)
    return buffer

st.subheader("📚 Step 1: Define the Concept")
reference_text = st.text_area(
    "Enter the correct reference explanation here:", 
    "Cloud computing is the delivery of computing services—including servers, storage, databases, networking, software, analytics, and intelligence—over the internet."
)

st.subheader("🎙️ Step 2:  Upload Your Audio")
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    temp_file_path = "temp_audio.wav"
    
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    with st.spinner("AI is transcribing the audio..."):
        result = model.transcribe(temp_file_path)
        student_text = result["text"]
        
    st.success("Transcription Complete!")
    st.info(f"**Student said:** {student_text}")
    
  
    

    st.subheader("📊 : Evaluation Results")
    with st.spinner("Analyzing conceptual understanding..."):
        ref_embedding = sbert_model.encode([reference_text])
        student_embedding = sbert_model.encode([student_text])
        similarity_score = cosine_similarity(ref_embedding, student_embedding)[0][0]
        final_score = round(similarity_score * 100, 2)
        
    st.metric(label="Conceptual Similarity Score", value=f"{final_score}%")
    
    if final_score > 75:
        st.success("Strong Understanding! 🌟")
    elif final_score > 50:
        st.warning("Moderate Understanding. Missed some key points. 🤔")
    else:
        st.error("Poor Understanding. Needs review. ❌")
        
    st.subheader("📈 : Speech Fluency")
    with st.spinner("Extracting audio features..."):
        filler_count = count_filler_words(student_text)
        st.write(f"**Filler Words Detected (um, uh, like):** {filler_count}")
        if filler_count > 3:
            st.warning("Note: High use of filler words. Try to pause silently instead.")
        
        y, sr = librosa.load(temp_file_path, sr=None)
        rms = librosa.feature.rms(y=y)[0]
        avg_energy = np.mean(rms)
        st.write(f"**Average Speech Energy:** {avg_energy:.4f}")
            
    st.write("**Audio Waveform Visualization:**")
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, color="#1f77b4")
    ax.set_title("Student Speech Waveform")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("🤖 : AI-Generated Feedback Report")
    
    # Pulled cleanly from secrets background file as we configured earlier
    api_key = st.secrets["GEMINI_API_KEY"]
    
    if st.button("Generate Final AI Report"):
        with st.spinner("Gemini is analyzing the student's performance..."):
            try:
                client = genai.Client(api_key=api_key)
                
                prompt = f"""
                Act as an expert educator evaluating a student's explanation of a technical concept.
                
                Reference Concept: {reference_text}
                Student's Spoken Answer: "{student_text}"
                Conceptual Accuracy Score: {final_score}%
                Filler Words Used: {filler_count}
                
                Please provide a short, encouraging feedback report containing:
                1. Two bullet points on what the student explained well.
                2. Two bullet points on areas for improvement.
                """
                
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                
                ai_report_text = response.text
                st.success("Report Generated Successfully!")
                st.write(ai_report_text)
                
                # Save report text dynamically into the user session state
                st.session_state["ai_report_text"] = ai_report_text
                
            except Exception as e:
                st.error(f"Error connecting to Gemini API: {e}")
                
    # Check if the AI report text exists in session memory to generate the PDF
    if "ai_report_text" in st.session_state:
        pdf_data = create_pdf_report(
            reference=reference_text,
            transcription=student_text,
            score=final_score,
            fillers=filler_count,
            energy=avg_energy,
            ai_feedback=st.session_state["ai_report_text"]
        )
        
        st.write("📥 **Export Evaluation:**")
        st.download_button(
            label="Download Structured PDF Report",
            data=pdf_data,
            file_name="VBCUA_Evaluation_Report.pdf",
            mime="application/pdf"
        )
            
    # Clean up
    os.remove(temp_file_path)