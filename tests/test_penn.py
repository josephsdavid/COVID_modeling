from comodels import penn
import numpy as np


def test_rolling_sum():
    a = np.array([1, 2, 3, 4, 5])
    window = 2
    expected = [3, 5, 7, 9]
    assert penn.rolling_sum(a, window).tolist() == expected
