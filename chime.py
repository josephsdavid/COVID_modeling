import numpy as np

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

    pars = generate_pars(S, infect_0, curr_hosp, hosp_rate, t_double,
                  contact_rate, hosp_share, hos_los, icu_los,
                  vent_los, R, t_rec, vent_rate, icu_rate)

    s, i, r = chime_sir(pars['S'], pars['I'], pars['R'],
                        pars['beta'], pars['gamma'], n_days, beta_decay)

    hosp, vent, icu = (pars[x]*i*pars['hosp_share'] for x in ['hosp_rate', 'vent_rate','icu_rate'])
    days = np.arange(0, n_days+1)
    data = dict(zip(['day','hosp','icu','vent'], [days, hosp, vent, icu]))
    return data


import matplotlib.pyplot as plt

tx = chime(28995881, 108, 20, n_days = 60)

import matplotlib.pyplot as plt

ax = plt.subplot()
for k in list(tx.keys())[1:]:
    ax.plot(tx[k], label = k)
    ax.legend()
plt.show()
