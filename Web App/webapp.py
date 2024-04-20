import streamlit as st
import pandas as pd
import base64
import requests
import time
from transformers import pipeline

# Function to load CSV or XLSX file
def load_file(file):
    if file is not None:
        try:
            df = pd.read_excel(file, engine='openpyxl')
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
    return None

# Function to fetch GitHub avatar and information
def fetch_github_info(username):
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Function to display GitHub contributors
def display_contributors():
    contributors = [
        "kapnishi", "VedantJoshi01", "atharva-mirkar", "bhushansansare",
        "karankherada", "ruddhi09", "saielshinde", "Sameer14122",
        "harsh1434", "RajdeepKrat", "burhanuddink"
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


# Function to download CSV file
def download_csv(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'data:file/csv;base64,{b64}'
    return href

# Function to add label to text
def add_label(classifier, texts, categories):
    label = []
    score = []
    for text in texts:
        start_time = time.time()
        output = classifier(text, categories, multi_label=False)
        end_time = time.time()
        time_taken = end_time - start_time
        max_score_index = output["scores"].index(max(output["scores"]))
        max_score = output['scores'][max_score_index]
        predicted_label = output["labels"][max_score_index]
        label.append(predicted_label)
        score.append(max_score)
        print("Time taken:", time_taken)
    return label, score

# Function to determine labels for preprocessed DataFrame
def determine_labels(preprocessed_df):
    texts = preprocessed_df['preprocessed_text']
    categories = ['ham', 'promotional', 'educational', 'financial', 'job', 'account verification', 'shopping', 
                  'rate experience', 'miscellaneous']
    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33")
    label, score = add_label(classifier, texts, categories)
    preprocessed_df['Label'] = label
    preprocessed_df['Score'] = score
    return preprocessed_df

# Sidebar
st.sidebar.title("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV or XLSX")

# Display contributors in the sidebar
display_contributors()

# Main screen
st.title("NLP Auto Labelling Project")
st.write("The NLP Auto Labelling Project is a web app that automates text labelling using advanced NLP models. Users upload CSV or XLSX files with unlabelled text, and the tool assigns labels automatically. It's perfect for tasks like categorizing customer feedback or classifying emails, streamlining data analysis.")

if uploaded_file is not None:
    # Load and display input file
    df = load_file(uploaded_file)
    if df is not None:
        st.subheader("Unlabelled Data")
        st.write(df)
        st.markdown("---")
        
        # Process the input data
        processed_df = determine_labels(df)
        
        # Display processed data
        st.subheader("Labelled Data")
        st.write(processed_df)
        
        # Download button for processed data
        st.download_button(label='Download', data=download_csv(processed_df), file_name='output.csv', mime='text/csv')