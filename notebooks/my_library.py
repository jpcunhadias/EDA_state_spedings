import nltk
import pandas as pd

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
from collections import Counter


# resources for this function: https://www.nltk.org/book/ch05.html

def get_most_common_nouns(df: pd.DataFrame,
                          column_name: str,
                          value: int) -> list:
    """
    The get_most_common_nouns function takes a dataframe and column name as input.
    It then extracts nouns from the text in that column, flattens the list of nouns into a single list,
    and counts the frequency of each noun. It returns this most common nouns as an output.

    :param df: Pass in the dataframe that contains the text
    :param column_name: Specify the column name of the text data
    :param value: Specify the number of most common nouns to return
    :return: A list of tuples containing the most common nouns and their frequencies
    :doc-author: Trelent
    """

    def extract_nouns(text: str) -> list:
        words = nltk.word_tokenize(text)
        tagged = pos_tag(words)

        #
        verbs = [word for word, pos in tagged if pos == 'NNP']
        return verbs

    # Apply the function to the text column
    df['nouns'] = df[column_name].apply(extract_nouns)

    # Flatten the list of verbs into a single list
    flat_list = [item for sublist in df['nouns'].tolist() for item in sublist]

    # Count the frequency of each verb
    nouns_counts = Counter(flat_list)

    # Get the most common nouns
    most_common_nouns = nouns_counts.most_common(value)

    return most_common_nouns
