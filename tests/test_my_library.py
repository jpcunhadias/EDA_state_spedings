import pandas as pd
from notebooks.my_library import get_most_common_nouns

# tests/test_my_library.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../notebooks')))


def test_get_most_common_nouns() -> None:
    text_list = ["John runs every morning.", "Mary walks in the park.", "Jane sings in the shower."]
    df = pd.DataFrame({'text': text_list})

    expected_result = []

    result = get_most_common_nouns(df, 'text')

    assert result == expected_result  # add assertion here
