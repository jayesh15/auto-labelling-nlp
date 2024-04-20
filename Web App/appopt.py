import streamlit as st
import base64
import time
import requests
from transformers import pipeline

# Function to add label to text
def add_label(classifier, text, categories):
    output = classifier(text, categories, multi_label=False)
    max_score_index = output["scores"].index(max(output["scores"]))
    max_score = output['scores'][max_score_index]
    predicted_label = output["labels"][max_score_index]
    return predicted_label, max_score

# Function to display GitHub contributors
def display_contributors():
    contributors = [
        "kapnishi"
    ]
    st.sidebar.title("Contributors")
    for username in contributors:
        user_info = fetch_github_info(username)
        if user_info:
            # Create columns layout
            col1, col2 = st.sidebar.columns([1, 4])
            # Anchor tag for redirection to GitHub profile
            with col1:
                st.markdown(f'<a href="https://github.com/{username}" target="_blank"><img src="{user_info["avatar_url"]}" width="50"></a>', unsafe_allow_html=True)
            # Display name and username in the second column
            with col2:
                st.write(f"""
                **{user_info['name']}**
                <br>
                @{user_info['login']}
                """, unsafe_allow_html=True)
        else:
            st.sidebar.write(f"No info available for {username}")

# Function to fetch GitHub avatar and information
def fetch_github_info(username):
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Main screen
st.title("NLP Auto Labelling Project")

# Sidebar
st.sidebar.title("Project Description")
st.sidebar.write("This project utilizes natural language processing techniques to automatically label text data into predefined categories.")
display_contributors()

# Main Screen
st.header("Input")
text_input = st.text_area("Enter your text here:")

if st.button("Submit"):
    if text_input:
        categories = ['ham', 'promotional', 'educational', 'financial', 'job', 'account verification', 'shopping', 
                      'rate experience', 'miscellaneous']
        classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33")

        # Process the input text
        label, score = add_label(classifier, text_input, categories)

        # Display processed data
        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader("Label")
            st.write(label)
        with col2:
            st.subheader("Score")
            st.metric("Score", score * 100, "%")
