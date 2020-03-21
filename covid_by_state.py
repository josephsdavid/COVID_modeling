import math
from comodels import Penn
from cotools import get_hopkins
import pandas as pd
import numpy as np
from states import states
from sklearn.linear_model import LinearRegression

r_naught = 2.2
social_distancing = 0
t_recovery = 20

help(Penn)

census = pd.read_csv("data/census.csv")

covid_numbers = pd.read_csv("http://coronavirusapi.com/time_series.csv")


covid_numbers

def get_us(d: dict) -> dict:
    idx = [i for i in range(len(d['Country/Region'])) if d['Country/Region'][i] == 'US']
    return {k:np.array(v)[idx] for k, v in d.items()}


def get_state_unabbrev(x: str) -> str:
    return states[x]

def get_state(x:str) -> str:
    return x[-2:].upper()



def agg_by_state(d: dict) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(d)
    df['Province/State'] = df['Province/State'].apply(get_state)
    out = df.groupby('Province/State').sum().copy()
    out = out.drop(['Lat', 'Long'], axis=1)
    out['State'] = out.index
    return out


us_co , us_dead, us_rec= (agg_by_state(get_us(x)) for x in get_hopkins())

np.arange(us_co.sum(0).shape[0])

(us_co.sum(0).apply(math.log))

def get_slope(X):
    lm = LinearRegression()
    lm.fit(np.arange(X.shape[0]).reshape(-1,1), X.apply(math.log))
    return lm.coef_[0]

get_slope(us_co.sum(0).drop('State'))

out = []
for i in range(us_co.shape[0]):
    data = us_co.iloc[i,:].drop('State')
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    out += [get_slope(data)]



us_co.sum(0)

deaths = us_dead.sum(1)

us_dead.keys()



deaths_today = deaths - us_dead.iloc[:,:-13].sum(1)


deaths_today['TX']

deaths_today['tx'.upper()]


