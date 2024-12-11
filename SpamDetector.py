import pickle
import streamlit as st
from gtts import gTTS  
import io
import os
import base64

language = 'en'

model = pickle.load(open('spam123.pkl', 'rb'))
cv = pickle.load(open('vec123.pkl', 'rb'))

def get_audio_bytes(filename):
    with open(filename, "rb") as f:
        audio_bytes = f.read()
    return base64.b64encode(audio_bytes).decode("utf-8")

st.title("Email Spam Classification Application")
st.write("This is a Machine Learning application to classify emails.")
st.subheader("Classification")

user_input = st.text_area("Enter an email to classify", height=250)

if st.button("Classify"):
    if user_input:
        data = [user_input]
        vec = cv.transform(data).toarray()
        result = model.predict(vec)

        if result[0] == 0:  
            st.success("This is Not A Spam Email")
            tts = gTTS(text="This is not a Spam Email", lang=language)
        else:
            st.error("This is A Spam Email")
            tts = gTTS(text="This is a Spam Email", lang=language)
        
        audio_filename = "output.mp3"
        tts.save(audio_filename)
        
        # Get the base64 encoded audio data
        audio_base64 = get_audio_bytes(audio_filename)
        
        # HTML for autoplay audio (using base64)
        audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
        """
        
        # Display the audio player with autoplay
        st.markdown(audio_html, unsafe_allow_html=True)
        
        # Optionally, delete the file after playing to avoid clutter
        os.remove(audio_filename)
    else:
        st.write("Please enter an email to classify.")
