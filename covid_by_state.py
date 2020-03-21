import numpy as np
import math
from comodels import PennDeath
from comodels.utils import states
from cotools import get_hopkins
import pandas as pd
from pprint import pprint
from sklearn.linear_model import LinearRegression
census = pd.read_csv("data/census.csv")
pops = census[['NAME', 'POPESTIMATE2019']]

nms = pops['NAME']


conf['Province/State']


social_distancing = 0
# I think this is fairly accurate
t_recovery = 23

# get the states out of the hopkins time series
def get_state_level(d: dict) -> dict:
    idx = [i for i in range(len(d['Province/State'])) if d['Province/State'][i] in states.values()]
    return {k:np.array(v)[idx] for k, v in d.items()}
conf, dead, rec = (pd.DataFrame.from_dict(get_state_level(x)).drop(['Lat','Long','Country/Region'], axis=1) for x in get_hopkins())


# get the growth rate from the data
def get_slope(X):
    lm = LinearRegression()
    lm.fit(np.arange(X.shape[0]).reshape(-1,1), X.apply(lambda x: math.log(x) if x != 0 else x))
    return lm.coef_[0]

# the zeros mess up our slope
us = conf.sum(0).drop('Province/State').copy()
us = us.loc[us!=0]

# 2 = (1 + growth_rate)**t
# log2(2) = log2(1+ growth_rate)**t
# 1 = t*(log2(1+growth_rate))
growth_rate = get_slope(us)
t_double = 1/(np.log2(1+growth_rate))
print(t_double)

# make it a function for later on
def doubling_time(gr: float) -> float:
    return 1/np.log2(1+gr)


state_growths = []
for i in range(conf.shape[0]):
    data = conf.iloc[i,:].drop('Province/State')
    data = data.loc[data!=0]
    # balance the growth rate towards the aggregate growth rate, as suggested by
    # Pat
    state_growths += [(get_slope(data) + growth_rate)/2]

# in general, things are similar to the aggregate rate, but a bit more
# pessimistic
print(sum(state_growths) / len(state_growths) - growth_rate)

state_growths = dict(zip(conf['Province/State'], state_growths))
state_d_times = {k: doubling_time(v) for k, v in state_growths.items()}


help()

