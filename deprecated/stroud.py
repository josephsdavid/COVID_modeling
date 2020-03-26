# Core Library modules
from typing import List

# Third party modules
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # cheating but fine

url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
countries = ["China", "Italy", "US", "Italy", "Spain", "Korea, South", "Iran"]


def get_over_100(url: str, countries: List[str] = countries) -> dict:
    data = pd.read_csv(url)
    data = data[data["Country/Region"].isin(countries)].groupby("Country/Region").sum()
    out = {}
    for idx, row in data.iterrows():
        out[idx] = np.array(row.drop(["Lat", "Long"]).tolist())
        out[idx] = out[idx][np.where(out[idx] >= 100)]
    return out


def get_over_100_cumsum(url: str, countries: List[str] = countries) -> dict:
    data = pd.read_csv(url)
    data = data[data["Country/Region"].isin(countries)].groupby("Country/Region").sum()
    out = {}
    for idx, row in data.iterrows():
        out[idx] = np.cumsum(np.array(row.drop(["Lat", "Long"]).tolist()))
        out[idx] = out[idx][np.where(out[idx] >= 100)]
    return out


countries = get_over_100(url)

mpl.use("tkagg")
plt.style.use("fivethirtyeight")

ax = plt.subplot()
for c, v in countries.items():
    ax.plot(v, label=c)
    ax.legend()
ax.set_title("daily infections after the first day with 100 confirmed cases")
plt.show()
