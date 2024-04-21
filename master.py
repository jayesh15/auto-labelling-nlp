import pandas as pd
import base64
import time
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string


# Function to load CSV or XLSX file from path or URL
def load_file(file_path_or_url):
    try:
        if file_path_or_url.startswith('http'):
            df = pd.read_csv(file_path_or_url, encoding='ISO-8859-1')  # Specify ISO-8859-1 encoding for CSV files
        else:
            df = pd.read_excel(file_path_or_url)
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

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

# Function to preprocess text using lemmatization
def preprocessing(df):
    # Download necessary NLTK resources
    nltk.download('stopwords')
    nltk.download('wordnet')
    
    # Exploratory Data Analysis (EDA)
    print("EDA")
    print("************"*3)
    # Printing DataFrame info
    print("INFO")
    print(df.info)
    print("-------------"*3)
    
    # checking for missing values
    print("NULL VALUES")
    print(df.isnull().sum())
    print("-------------"*3)
    
    # checking for duplicate values
    print("DUPLICATE VALUES")
    print("number of duplicate values")
    num = df.duplicated().sum()
    print(num)
    # If duplicate values are found removing them
    if num > 0:
        print("shape of df before dropping")
        print(df.shape)
        df = df.drop_duplicates(keep='first')
        print("shape of df before dropping")
        print(df.shape)
        
    # Replacing unnecessary letters/words
    replace_list = [" &lt;#&gt;","Ã","&gt;","\\n","\\t","\\r","â","€","™","ð","Ÿ","ðŸ‘","$","â€™ll","ƒ","¢","â€ƒ","â€¢","Â§","§","Â","Ã¼","Ã","¼","º","œ","˜","£","â€“","â€œ","&lt;#&gt;","â€Œ","ðŸŽ","ð","Ÿ","Ž","Í","â€Œ ï»¿ Í","ï»¿","ðŸ”¨","ðŸ¤©","©","¤","±","ðŸ˜±","ðŸ˜±"," Í â€Œ ï»¿","ÿ","x = = x","*"]
    for letter in replace_list:
        # Replacing each unnecessary words/letter with an empty string
        df[df.columns[0]] = df[df.columns[0]].str.replace(letter, '')
    
    # List of email-related words to replace
    email_replace_list = ["Subject:","cc:"]
    for letter in email_replace_list:
        # Replacing email-related word with an empty string
        df[df.columns[0]] = df[df.columns[0]].str.replace(letter, '')
    
    # Initialize the WordNet Lemmatizer
    lemmatizer = WordNetLemmatizer()
    df['preprocessed_text'] = df[df.columns[0]].apply(lambda x: preprocessing_text(x, lemmatizer))
    
    return df

# Function to preprocess text using lemmatization
def preprocessing_text(text, lemmatizer):
    # Tokenize the text into words
    text = nltk.word_tokenize(text)

    # Filtering out non-alphanumeric words
    filtered_words = []
    for word in text:
        if word.isalnum():
            filtered_words.append(word)

    # Updating the 'text' variable with the filtered words
    text = filtered_words[:]
    
    #Empty list for lemmatized word
    lemmatized_words= []
    for word, pos in nltk.pos_tag(text):
        #Checking if the word is alphanumeric and not in stopwords or punctuation
        if word.isalnum() and word not in stopwords.words('english') and word not in string.punctuation:
            # Get the first letter of the POS tag and converting it to lower
            pos = pos[0].lower()
            # Consider only adjectives, nouns, and verbs for lemmatization
            if pos in {'a', 'n', 'v'}:
                lemmatized_words.append(lemmatizer.lemmatize(word, pos))
            else:
                # If not in a,n,v then append the original word
                lemmatized_words.append(word)

    # Return the preprocessed text as a single string
    return " ".join(lemmatized_words)

# Main function
def main():
    # Get input CSV path or URL from user
    file_path_or_url = input("Enter the path or URL of the input CSV file: ")
    
    # Load input file
    df = load_file(file_path_or_url)
    
    # If file loaded successfully, proceed
    if df is not None:
        # Preprocess the input data
        df = preprocessing(df)
        
        # Process the input data
        processed_df = determine_labels(df)
        
        # Display processed data
        print("Labelled Data:")
        print(processed_df)

# Execute main function
if __name__ == "__main__":
    main()