# Importing necessary libraries
import time
import pandas as pd
from transformers import pipeline


def add_label(classifier, texts, categories):
    
    # Initialize lists to store label and score
    label = []
    score = []
    
    for text in texts:

        # start time
        start_time = time.time()
        # Predict label for for the current text
        output = classifier(text, categories, multi_label=False)

        # Record end time and calculate the time taken 
        end_time = time.time()
        time_taken = end_time - start_time

        # Get the index of the label with the highest score
        max_score_index = output["scores"].index(max(output["scores"]))
        max_score = output['scores'][max_score_index]

        # Get the label with the highest score
        predicted_label = output["labels"][max_score_index]

        # Append predicted label and score to respective lists
        label.append(predicted_label)
        score.append(max_score)

        print("Time taken:", time_taken)
    
    #Return list of predicted labels and scores
    return label,score


def determine_labels(preprocessed_df):
    
    # Extract texts to process
    texts = preprocessed_data['preprocessed_text']
    
    # Define the categories
    categories = ['ham', 'promotional', 'educational', 'financial', 'job', 'account verification', 'shopping', 
                  'rate experience', 'miscellaneous']
    
    # Initialize the zero-shot classification pipeline
    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33")
    
    # Add labels and score to text
    label,score = add_label(classifier, texts, categories)
    
    preprocessed_df['Label'] = label
    preprocessed_df['Score'] = score
    
    return preprocessed_df