# Importing necessary libraries
import pandas as pd
import nltk
import io
import requests
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string


# Downloading resources
def download():
    nltk.download('stopwords')
    nltk.download('wordnet')
    

def eda(df):
    # Exploratory Data Analysis (EDA) begins
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
    if num>0:
        print("shape of df before dropping")
        print(df.shape)
        df = df.drop_duplicates(keep='first')
        print("shape of df before dropping")
        print(df.shape)
        
        
def replace_words(df):
    # replacing unnecessary letters/words
    replace_list = [" &lt;#&gt;","Ã","&gt;","\\n","\\t","\\r","â","€","™","ð","Ÿ","ðŸ‘","$","â€™ll","ƒ","¢","â€ƒ","â€¢","Â§","§","Â","Ã¼","Ã","¼","º","œ","˜","£","â€“","â€œ","&lt;#&gt;","â€Œ","ðŸŽ","ð","Ÿ","Ž","Í","â€Œ ï»¿ Í","ï»¿","ðŸ”¨","ðŸ¤©","©","¤","±","ðŸ˜±","ðŸ˜±"," Í â€Œ ï»¿","ÿ","x = = x","*"]
    for letter in replace_list:
        # Replacing each unnecessary words/letter with an empty string
        df['Message'] = df['Message'].str.replace(letter, '')
    return df


def replace_email_words(df):
    #List of email-related words to replace
    email_replace_list = ["Subject:","cc:"]
    for letter in email_replace_list:
        # Replacing email-related word with an empty string
        df['Message'] = df['Message'].str.replace(letter, '')
    return df


def preprocessing_text(text):
    # Initialize the WordNet Lemmatizer
    lemmatizer = WordNetLemmatizer()
    # Checking if the input text is a string
    if isinstance(text, str):
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
    else:
        # Convert non-string types to string
        return str(text)
    
    
def preprocessing(df):
    
    # Calling the download() function
    download()
    
    # performing EDA and basic preprocessing
    eda(df)
    df = replace_words(df)
    df = replace_email_words(df)
    
    # preprocess text using lemmatisation
    df['preprocessed_text'] = df['Message'].apply(lambda x: preprocessing_text(x))
    
    return df