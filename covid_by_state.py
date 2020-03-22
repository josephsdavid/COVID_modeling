import numpy as np
import math
from comodels import PennDeath
from comodels.utils import states
from cotools import get_hopkins
import pandas as pd
from pprint import pprint
from sklearn.linear_model import LinearRegression
census = pd.read_csv("census.csv")
pops = census[['NAME', 'POPESTIMATE2019']]
nms = pops['NAME']


social_distancing = 0
# I think this is fairly accurate
t_recovery = 23


# get the states out of the hopkins time series

def get_state_level(d: dict) -> dict:
    idx = [i for i in range(len(d['Province/State'])) if d['Province/State'][i] in states.values()]
    return {k:np.array(v)[idx] for k, v in d.items()}
conf, dead, rec = (pd.DataFrame.from_dict(get_state_level(x)).drop(['Lat','Long','Country/Region'], axis=1) for x in get_hopkins())


# get the growth rate from the data
def get_slope(X: pd.DataFrame) -> float:
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
# print(t_double)


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
# print(sum(state_growths) / len(state_growths) - growth_rate)

state_growths = dict(zip(conf['Province/State'], state_growths))
d_times = {k: doubling_time(v) for k, v in state_growths.items()}
d_times[nms[0]] = t_double

# something something something


# TODO: make plotly code that does https://plot.ly/python/sliders/ but for our
# output. Should be cake, maybe a massive loop of all the states, byt ezpz

# we are going to assume D_today =  0 for now, for the sake of not embarassing
# ourselves, as per Niels' suggestion. D_today is not actually like a crucial
# paremeter, it is a tool for estimating where we are on the curve. Code is not
# fully trustworthy yet involving that, so just set it to be zero and be naive
# (MVP)

names = [x for x in nms if x in d_times.keys()]



new_names = {i:v for i, v in enumerate(conf['Province/State'])}
c_tot, d_tot, r_tot = (x.sum(1).rename(new_names) for x in [conf, dead, rec])
c_tot['United States'] = c_tot.sum()
d_tot['United States'] = d_tot.sum()
r_tot['United States'] = r_tot.sum()

pops[pops['NAME'] == 'United States']['POPESTIMATE2019'].values[0]


out = {}
sds = [0, 0.2, 0.5, 0.7]
for n in names:
    state_curve = {}
    for s in sds:
        N = pops[pops['NAME'] == n]['POPESTIMATE2019'].values[0]
        I = c_tot[n]
        R = r_tot[n]
        D = r_tot[n]
        td = d_times[n]
        model = PennDeath(N, I, R, D, 0, t_double=td, recover_time=t_recovery)
        curve, occ = model.sir(60)
        sir = {k:v for k, v in curve.items() if k in ['susceptible', 'infected', 'recovered']}
        hosp_use = {k:v for k, v in curve.items() if k not in ['susceptible', 'infected', 'recovered']}
        state_curve[s] = {'SIR': sir, 'Hospital Use': hosp_use, 'Hospital Census': occ}
    out[n] = state_curve
