import nltk
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

from typing import List, Tuple
from collections import Counter
from datetime import datetime

stop_words = set(nltk.corpus.stopwords.words('portuguese'))


def preprocess_text_pipeline(text: str) -> str:
    """
    The preprocess_text_pipeline function takes a string as input and returns a new string
    as output. The input is lowercase, tokenized, and filtered of stop words. The output
    is the processed text.

    :param text:str: Pass the text to be preprocessed
    :return: A string of tokens that have been filtered for stop words and made lowercase
    :doc-author: Trelent
    """

    text = text.lower()

    tokens = nltk.word_tokenize(text)

    filtered_tokens = [w for w in tokens if not w in stop_words]

    return ' '.join(filtered_tokens)


# resources for this function: https://www.nltk.org/book/ch05.html
def get_most_common_nouns(df: pd.DataFrame,
                          column_name: str,
                          value: int) -> list:
    """
        The get_most_common_nouns function takes a dataframe and column name as input.
        It then extracts nouns from the text in that column, flattens the list of nouns into a single list,
        and counts the frequency of each noun. It returns this most common nouns as an output.

        :param df:pd.DataFrame: Specify the dataframe that contains the text
        :param column_name:str: Specify the column name of the text data
        :param value:int: Specify the number of most common nouns to return
        :return: A list of tuples, where the first element in each tuple is a noun and the second element is its frequency
        :doc-author: Trelent
    """

    def extract_nouns(text: str) -> list:
        words = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(words)

        nouns = [word for word, pos in tagged if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
        return nouns

    # Apply the function to the text column
    df['nouns'] = df[column_name].apply(extract_nouns)

    # Flatten the list of nouns into a single list
    flat_list = [item for sublist in df['nouns'].tolist() for item in sublist]

    # Count the frequency of each noun
    nouns_counts = Counter(flat_list)

    # Get the most common nouns
    most_common_nouns = nouns_counts.most_common(value)

    return most_common_nouns


def calculate_sample_size(population_size: int,
                          margin_of_error: float,
                          confidence_level: float,
                          variability: float = None
                          ) -> int:
    """
    The sample_size function takes a population size, margin of error, and confidence level as input.
    It then calculates the sample size needed to achieve the desired margin of error and confidence level.
    The sample size is returned as an output. The variability parameter is optional, and it can be used to
    specify the variability of the population. If the variability is not specified, then it is assumed to be 0.5.

    The population size is the total number of items in the population. It's not used in the calculation, but it's
    included as an input parameter for completeness.

    In the context of this project, the variability can be the proportion of an item divided by
    the total number of items if you want to calculate the sample size for a proportion. For example,
    if the proportion of items that are red is 0.3, then the variability is 1- 0.3 = 0.7.

    :param population_size: int: Specify the total number of items in the population
    :param margin_of_error: float: Specify the desired margin of error
    :param confidence_level: float: Specify the desired confidence level
    :param variability: float: Specify the variability of the population
    :return: sample_size: int: The sample size needed to achieve the desired margin of error and confidence level
    """

    Z = 0.0
    p = 0.5
    if variability is not None:
        p = 1 - variability

    Z = abs(stats.norm.ppf((1 - confidence_level) / 2))
    n = (Z ** 2 * p * (1 - p)) / (margin_of_error ** 2)
    return int(n)


def sample_df(df: pd.DataFrame,
              percent: float) -> pd.DataFrame:
    """
    The sample_df function takes a DataFrame and returns a new DataFrame that contains
    a random sample of the original. The percentage of rows to return is determined by
    the percent parameter. For example, if percent=0.5, then 50% of the rows will be returned.

    :param df:pd.DataFrame: Specify the dataframe that will be sampled
    :param percent:float: Specify the percentage of rows to sample
    :return: A random sample of rows from the input dataframe
    :doc-author: Trelent
    """

    # Determine the number of rows in the dataframe
    n_rows = df.shape[0]

    # Calculate the number of rows to sample based on the desired percentage
    # sample_size = calculate_sample_size(population_size=n_rows, margin_of_error=0.05, confidence_level=0.95)

    # for a specific percentage of the rows, uncomment the line below
    sample_size = int(n_rows * percent)

    # Use numpy randint function to generate random indices
    rand_indices = np.random.randint(0, n_rows, size=sample_size)

    # Use the random indices to index into the dataframe
    random_df = df.iloc[rand_indices]

    return random_df


def calculate_working_days(start_date: datetime,
                           end_date: datetime) -> int:
    """
    The calculate_working_days function takes a start date and end date as input.
    It then calculates the number of working days between the start date and end date.
    The number of working days is returned as an output.

    :param start_date: datetime: Specify the start date
    :param end_date: datetime: Specify the end date
    :return: working_days: int: The number of working days between the start date and end date
    """

    # Convert start_date and end_date to pandas datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Calculate the difference between the dates
    difference = end_date - start_date

    # Generate a range of dates between the start and end dates
    dates = pd.date_range(start=start_date, end=end_date, freq='B')

    # Return the number of working days as the length of the dates range
    return len(dates)


def plot_boxplot_for_stems(nouns_and_counts: list[Tuple[str, int]],
                           root_nouns_to_include: list) -> None:
    # NOT FUNCTIONAL
    """
    The plot_boxplot_for_stems function creates a boxplot of the counts for each root noun in the list
    of nouns and their counts. The function takes two arguments:
        -nouns_and_counts: A list of tuples, where each tuple contains a noun and its count.
        -root_nouns: A list of strings representing all the unique roots in the given corpus.

    :param nouns_and_counts: Pass in a list of tuples
    :param root_nouns_to_include:list: Specify the list of root nouns to plot
    :return: None
    :doc-author: Trelent
    """

    stemmer = nltk.stem.PorterStemmer()
    root_nouns = {}
    for noun, count in nouns_and_counts:
        root = stemmer.stem(noun)
        if root in root_nouns:
            root_nouns[root] += count
        else:
            root_nouns[root] = count

    filtered_root_nouns = {
        root: count for root, count in root_nouns.items() if root in root_nouns_to_include
    }

    fig, ax = plt.subplots()
    ax.boxplot(list(filtered_root_nouns.values()))
    ax.set_xticklabels(list(filtered_root_nouns.keys()))

    plt.show()


def plot_bar_chart_for_filtered_df(df: pd.DataFrame,
                                   filter_condition: bool,
                                   column: str) -> None:
    """
    The plot_bar_chart_for_filtered_df function plots a bar chart for the filtered dataframe.
    It takes in a dataframe, filter condition and column as input parameters. It returns None.

    :param df:pd.DataFrame: Pass in the dataframe that we want to plot
    :param filter_condition:bool: Filter the dataframe
    :param column:str: Specify the column that we want to plot
    :return: None
    :doc-author: Trelent
    """
    values_counted_df = df[filter_condition][column].value_counts().head()
    df_column_size = df[filter_condition][column].size

    labels = list(values_counted_df.keys())
    values = [(v / df_column_size) * 100 for v in values_counted_df.values]

    fig, ax = plt.subplots()
    ax.bar(labels, values)

    ax.set_title(f'Bar Chart for {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Count (%)')

    plt.show()


def get_season(date: datetime) -> str:
    """
    The get_season function takes a date as input and returns the season of the year as a string.
    The function takes a date as input and returns the season of the year as a string.

    :param date: datetime: Specify the date
    :return: season: str: The season of the year
    """

    # Convert the date to a pandas datetime object
    date = pd.to_datetime(date)

    # Determine the month of the year
    month = date.month

    # Determine the season of the year based on the month
    if month in [12, 1, 2]:
        season = 'Verão'
    elif month in [3, 4, 5]:
        season = 'Outono'
    elif month in [6, 7, 8]:
        season = 'Inverno'
    else:
        season = 'Primavera'

    return season
