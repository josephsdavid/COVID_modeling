import numpy as np
import pandas as pd

def sir_step (S, I , R, beta, gamma, N):
    Sn = (-beta *S*I) + S
    In = (beta*S*I - gamma * I) + I
    Rn = gamma*I + R
    Sn, Rn, In = (0 if x <0 else x for x in [Sn, Rn, In])

    scale = N / (Sn + In + Rn)
    return (x*scale for x in [Sn, Rn, In])


def chime_sir(S, I, R, beta, gamma, n_days, beta_decay = None):
    N = sum([S, I, R])
    s, i, r = ([x] for x in  [S, I, R])

    for _ in range(n_days):
        S, I, R = sir_step(S, I, R, beta, gamma, N)
        beta = beta*(1-beta_decay) if beta_decay is not None else beta
        s += [S]
        i += [I]
        r += [R]

    return (np.array(x) for x in [s,i,r])


def generate_pars(S, infect_0, curr_hosp, hosp_rate, t_double,
                  contact_rate, hosp_share, hos_los, icu_los,
                  vent_los, R, t_rec, vent_rate, icu_rate):
    out = {}
    out['S'] = S
    out['infection_known'] = infect_0
    out['hosp_rate'] = hosp_rate
    out['vent_rate'] = vent_rate
    out['icu_rate'] = icu_rate
    out['hosp_los'] = hos_los
    out['icu_los'] = icu_los
    out['vent_los'] = vent_los
    out['hosp_share'] = hosp_share
    infect_total = curr_hosp / hosp_share / hosp_rate
    out['I'] = infect_total
    out['detect_prob'] = infect_0 / infect_total
    out['R'] = R
    out['growth_intrinsic'] = 2**(1/t_double)  - 1
    out['t_rec'] = t_rec
    out['gamma'] = 1 / t_rec
    out['contact_rate'] = contact_rate
    out['beta'] = ((out['growth_intrinsic'] + out['gamma']) / S) * (1-contact_rate)
    out['r_t'] = out['beta'] / out['gamma'] * S
    out['r_naught'] = out['r_t'] / (1-contact_rate)
    out['t_double_base'] = t_double
    out['t_double_true'] = 1/np.log2(out['beta']*S - out['gamma'] + 1)
    return out


def chime(S, infect_0, curr_hosp, hosp_rate=0.05, t_double=6,
          contact_rate=0, hosp_share = 1., hos_los=7, icu_los=9,
          vent_los=10, R = 0, t_rec = 14, beta_decay = None, n_days = 60,
          vent_rate=0.01, icu_rate=0.02):
    '''
    chime: SIR model for ventilators, ICU , and hospitilaizations.
    -----------------------------
    parameters:
        S: population size
        infect_0: number of confirmed infections
        curr_hosp: number of currently hospitalized patients
        hosp_rate: hospitalization_rate = 0.05
        t_double: time to double without distancing = 6
        contact_rate: reduction in contact due to distancing = 0
        hosp_share: proportion of the population represented by hospitalization data = 1
        hos_los: how long people stay in the hospital = 7
        icu_los : how long people stay in the ice = 9
        vent_los : how long people stay with ventilators = 10
        R : number recovered present data = 0
        t_rec : number of days to recover = 14
        betay_decay : beta decay = None
        n_days: number of days ahead to look = 60
        vent_rate: rate of people who need ventilators = 0.01
        icu_rate: rate of people who need icu = 0.02
    '''

    pars = generate_pars(S, infect_0, curr_hosp, hosp_rate, t_double,
                  contact_rate, hosp_share, hos_los, icu_los,
                  vent_los, R, t_rec, vent_rate, icu_rate)

    s, i, r = chime_sir(pars['S'], pars['I'], pars['R'],
                        pars['beta'], pars['gamma'], n_days, beta_decay)

    hosp, vent, icu = (pars[x]*i*pars['hosp_share'] for x in ['hosp_rate', 'vent_rate','icu_rate'])
    # something is wrong in general, but will be up and running by bedtime
    days = np.arange(0, n_days+1)
    data = dict(zip(['day','hosp','icu','vent'], [days, hosp, icu, vent]))
    proj = pd.DataFrame.from_dict(data)
    proj_admits = proj.iloc[:-1,:] - proj.shift(1)
    proj_admits[proj_admits <0 ] = 0
    plot_days = n_days - 10
    proj_admits["day"] = range(proj_admits.shape[0])
    los_dict = {k[:-4]: pars[k] for k in ['hosp_los', 'icu_los', 'vent_los']}
    census_dict = {}
    for k, los in los_dict.items():
        census = (
            proj_admits.cumsum().iloc[:-los,:] - proj_admits.cumsum().shift(los).fillna(0)
        ).apply(np.ceil)
        census_dict[k] = census[k]
    census_df = pd.DataFrame(census_dict)
    census_df["day"] = census_df.index
    census_df = census_df[["day", "hosp", "icu", "vent"]]

    census_table = census_df[np.mod(census_df.index, 7) == 0].copy()
    census_table.index = range(census_table.shape[0])
    census_table.loc[0, :] = 0
    census_table = census_table.dropna().astype(int)
    return data, proj_admits, census_table


dtx, admits, census = chime(1345076, 35, 2, n_days = 140)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')

plt.plot(list(range(5)))
plt.show()
