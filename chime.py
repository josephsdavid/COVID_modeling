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


def generate_pars(S, infect_0, curr_hosp, hosp_rate=0.05, t_double=6,
                  contact_rate=0, hosp_share = .15, hos_los=7, icu_los=9,
                  vent_los=10, R = 0, t_rec = 14):
    out = {}
    out['S'] = S
    out['infection_known'] = infect_0
    out['hosp_los'] = hos_los
    out['icu_los'] = icu_los
    out['vent_los'] = vent_los
    out['hosp_share'] = hosp_share
    infect_total = curr_hosp / market_share / market_rate
    out['I'] = infect_total
    out['detect_prob'] = infect_0 / infect_total
