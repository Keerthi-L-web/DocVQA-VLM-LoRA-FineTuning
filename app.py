import streamlit as st
import pandas as pd
from PIL import Image
import tempfile
import os
import json
from datetime import datetime

# Streamlit Page Config
st.set_page_config(page_title="DocVQA LoRA Pro", page_icon="📄", layout="wide")

# Only load inferencer when needed to save resources and allow UI to render quickly
@st.cache_resource
def load_inferencer():
    try:
        from src.inference import DocVQAInferencer
        return DocVQAInferencer()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    st.title("📄 Document Visual Question Answering")
    st.markdown("Upload a document image and ask a question to get AI-predicted answers using **BLIP + LoRA**.")
    
    # Initialize session state for history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("1. Upload Document")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Document", use_column_width=True)
            
            # Save uploaded file temporarily for inference
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                image.save(tmp.name)
                tmp_path = tmp.name

    with col2:
        st.header("2. Ask Question")
        question = st.text_input("Enter your question about the document:")
        
        if st.button("Predict Answer", type="primary"):
            if uploaded_file is None:
                st.warning("Please upload a document first.")
            elif not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Analyzing document..."):
                    inferencer = load_inferencer()
                    if inferencer:
                        try:
                            answer, confidence = inferencer.predict(tmp_path, question)
                            st.success("Analysis Complete!")
                            
                            st.markdown(f"### Predicted Answer: **{answer}**")
                            st.info(f"Confidence Score: {confidence:.2%}")
                            
                            # Add to history
                            st.session_state.history.append({
                                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Filename": uploaded_file.name,
                                "Question": question,
                                "Answer": answer,
                                "Confidence": confidence
                            })
                            
                        except Exception as e:
                            st.error(f"Prediction failed: {str(e)}")
                            
                # Cleanup temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass

    # History Section
    st.divider()
    st.header("🕒 Prediction History")
    
    if st.session_state.history:
        df_history = pd.DataFrame(st.session_state.history)
        st.dataframe(df_history, use_container_width=True)
        
        # Download Results
        csv = df_history.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download History as CSV",
            data=csv,
            file_name='docvqa_history.csv',
            mime='text/csv',
        )
    else:
        st.info("No predictions made yet.")

if __name__ == "__main__":
    main()
