import io
import requests
import pandas as pd
from preprocessing import preprocessing
from auto_labelling import determine_labels

    #Reading the excel data
text_data_url = "https://raw.githubusercontent.com/jayesh15/auto-labelling-nlp/main/text_dataset.csv"
download = requests.get(text_data_url).content

# Reading the downloaded content and turning it into a pandas dataframe
df = pd.read_csv(io.StringIO(download.decode('ISO-8859-1')))


# Preprocessing the dataframe
preprocessed_df = preprocessing(df)
# Determining labels for preprocessed dataframe
labelled_df = determine_labels(preprocessed_df)


# Saving the labelled dataframe to csv file
labelled_df.to_csv(r'Downloads\labelled_text.csv')