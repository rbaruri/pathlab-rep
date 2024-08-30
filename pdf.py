import streamlit as st
from PIL import Image
from groq import Groq
import io
import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from ocr import process_document_sample, get_mime_type

load_dotenv(Path(".env"))

# App header and description
st.title("PathoAssist - Simplifying Pathology Reports for You 📄")

st.markdown(
    """
    _Welcome to PathoAssist! 
    This app simplifies your pathology report results by providing easy-to-understand explanations in plain language._ 
    """
)

st.info(
    """
    Simply upload a photo or PDF of your report to receive a clear summary. Additionally, you can chat with your report to ask specific questions and get detailed answers, helping you understand your results better.
    """
)

# File uploader for image or PDF selection
report_original = st.file_uploader("Select the photo or PDF of your report you want to interpret", type=['png', 'jpg', 'jpeg', 'pdf'])

# Define function to process file
@st.cache_data(show_spinner="Extracting text from the report...")
def process_file(file):
    try:
        file_extension = os.path.splitext(file.name)[-1]
        mime_type = get_mime_type(file_extension)
        file_content = io.BytesIO(file.read())

        # Perform OCR using your OCR module
        report_text = process_document_sample(
            project_id="tokyo-concept-417109",
            location="us",
            processor_id="d4996bc6d6ba788",
            file_content=file_content,
            mime_type=mime_type,
            processor_version_id="pretrained-ocr-v2.0-2023-06-02"
        )
    except Exception as e:
        st.error(f"Error processing file: {e}")
        report_text = None
    return report_text

# Initialize conversation history
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# LLM integration
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

# File submission logic
if report_original is not None:
    file_type = report_original.type
    
    if st.button("Submit", type="secondary"):
        if file_type in ['image/png', 'image/jpeg', 'image/jpg', 'application/pdf']:
            report_text = process_file(report_original)
        else:
            st.error("Unsupported file type.")
            report_text = None

        if report_text:
            with st.spinner("Summarizing the report..."):
                try:
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": """
                                    You are a system specialized in explaining medical and scientific terms in an accessible way to laypeople. Your mission is to receive a medical exam report and provide a concise explanation of the findings described for the patient, using simple language understandable to a primary school student.

                                    Please provide the explanation in the following structure:

                                    1.**Details**:
                                        - Take details of patients such as name, age, gender, test date, test type, doctor name etc. if available.

                                    2.**Summary of Results**: 
                                        - Summarize the result of the examination.
                                        - Highlight any abnormal findings. If there are none, state that nothing abnormal was identified.

                                    3. **Explanation of Severity**:
                                        - Clearly and simply explain the severity of any abnormal findings in detail.
                                        - Emphasize that the final word should always come from the responsible doctor.

                                    4. **Critical Findings (if applicable)**:
                                        - Highlight any critical findings that require immediate attention.
                                        - Emphasize urgency and encourage the patient to schedule a return consultation as soon as possible.

                                    """ + report_text
                            }
                        ],
                        model="llama3-8b-8192",
                    )
                    assistant_summary = chat_completion.choices[0].message.content
                    st.session_state.conversation.append({
                        "role": "system",
                        "content": assistant_summary
                    })
                    st.success("Summarization completed successfully ✅")
                    
                except Exception as e:
                    st.error(f"Error generating summary: {e}")

# Display the summary and then the search bar
if st.session_state.conversation:
    with st.expander("**Summary**", expanded=True):
        st.write(st.session_state.conversation[-1]['content'])

# User question input (enabled only after submission)
user_question = st.chat_input("Ask a question", disabled=not report_original)

if user_question:
    st.session_state.conversation.append({
        "role": "user",
        "content": user_question
    })

    with st.spinner("Generating response..."):
        try:
            chat_completion = client.chat.completions.create(
                messages=st.session_state.conversation + [
                    {
                        "role": "user",
                        "content": user_question
                    }
                ],
                model="llama3-8b-8192",
            )
            assistant_response = chat_completion.choices[0].message.content

            # Append assistant response to the conversation
            st.session_state.conversation.append({
                "role": "assistant",
                "content": assistant_response
            })
        except Exception as e:
            st.error(f"Error generating response: {e}")

    # Display the conversation history in a chatbot-like format
    for message in st.session_state.conversation:
        if message["role"] == "user":
            with st.chat_message("User"): 
                st.write(f"**You:** {message['content']}")
        elif message["role"] == "assistant":
            with st.chat_message("Assistant"):
                st.write(f"**Assistant:** {message['content']}")

# Clear chat button
if st.session_state.conversation:
    if st.button("Clear Chat", type="primary"):
        st.session_state.conversation = []
        st.experimental_rerun()
