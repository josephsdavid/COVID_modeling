import numpy as np
from .sir import SIR, SIRD

def calc_hosp_numbers(i: np.ndarray, rate: float) -> np.ndarray:
    """
    calc_hosp_numbers:
        given an infection curve i, and a rate, how many people will use
        resource XYZ
    Inputs:
        i: np.ndarray = infection curve
        rate: float = rate of whatever
    Outputs: np.ndarray
    """
    return i*rate


def calc_admission_deltas(admits: np.ndarray) -> np.ndarray:
    """
    calc_admission_deltas:
        given admissions to a hosp/icu/ventilators, what is the daily change in
        admissions?
    Inputs:
        admits: np.ndarray = number of admissions over time
    Outputs: np.ndarray
    """
    out = admits[1:] - admits[:-1]
    out[np.where(out < 0)] = 0
    return out


def rolling_sum(a: np.ndarray, window: int) -> np.ndarray:
    """
    rolling_sum:
        calculate the a rolling sum of a numpy array
    Inputs:
        a: np.ndarray = the array you want to calculate a rolling sum for
        window: int = the size of the window to sum for
    Outputs:
        np.ndarray
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    rolled = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return rolled.sum(-1).copy()


class Penn(SIR):
    """
    Penn:
        Make SIR predictions given the assumed detection rate (out of all infected cases), using
        the parameter estimation method from the Penn model.
    ----------------------------------------------------------------------
    Parameters:
        S: pop size, Number of people affected
        I: infected, Number of positive tests in region
        R: recovered, number of recoveries
        detect_prob: percentage of all infected cases in region which you believe are being detected = 1
        hosp_rate: rate of infected admitted to the hospital = 0.05
        icu_rate: rate of infected who need to be in ICU = 0.02
        vent_rate: rate of infected who need ventilators = 0.01
        contact_reduction: percent contact reduced by social distancing = 0
        t_double: time to double number of infected = 2.
        beta_decay: decay rate of beta, which represents how often a contact results in a new infection = 0
        vent_los: time one patient takes up a ventilator=10
        hos_los: time one patient takes up a normal hospital bed = 7
        icu_los: time one patient takes up an ICU bed = 9
        recover_time: time to get better, to shift from I to R = 14

    Attributes:
        rates: dict = rate of hospitalization, icu, and ventilators
        los: dict = length of stay at hospital, icu, and ventilator
        contact_reduction: float = contact reduction
        t_double: float = number of days to double
        intrinsic_growth: float = growth rate
        recover_time: float = time to recover from illness
        beta: float =  how often a contact results in a new infection
        gamma:  float = rate at which an infected person recovers
        beta_decay:  float = decay rate of beta
        r_naught:  float = speadability of disease
        r_t:  float = r_naught after distancing
        S: int = number of susceptible people
        I: int = number of actually infected people
        R: int = number of recovered people


    Methods:
        sir(n_days):
            run simulation, see docstring for sir
    """
    def __init__(self, S: int, I: int, R: int, detect_prob: float = 1,
                 hosp_rate: float=0.05, icu_rate: float=0.02,
                 vent_rate: float=0.01, contact_reduction: float=0.,
                 t_double: float=2, beta_decay: float=0, vent_los: float=10,
                 hos_los: float=7, icu_los: float=9, recover_time: float=14) -> None:
        self.rates = {'hospital': hosp_rate,'icu': icu_rate, 'ventilator':vent_rate}
        self.los = dict(zip(self.rates.keys(), [hos_los, icu_los, vent_los]))
        self.contact_reduction = contact_reduction
        self.t_double = t_double
        self.intrinsic_growth = 2**(1/t_double) - 1
        self.recover_time = recover_time
        gamma = 1/self.recover_time
        beta = ((self.intrinsic_growth + gamma) / S)  * (1-contact_reduction)
        self.r_t = beta/gamma * S
        self.r_naught = self.r_t / (1-contact_reduction)
        self.I = I / detect_prob
        super().__init__(S, self.I, R, beta, gamma, beta_decay)

    def sir(self, n_days: int) -> (dict, dict):
        """
        sir: fit SIR model to the data
        -----------------------------------
        Inputs:
            n_days: int = number of days ahead to predict
        Outputs:
            curve: dict :
                susceptible: np.ndarray = history of susceptible pop
                infected: np.ndarray = history of infection
                recovered: np.ndarray = history of recovery
            admissino: dict: of occupancy
                hospitalized
                icu
                ventilator
        """
        s, i, r = super().sir(n_days)
        out = {}
        admissions = {}
        out['infected'] = i
        out['recovered'] = r
        out['susceptible'] = s
        for k in self.rates.keys():
            # calculate raw numbers
            out[k] = calc_hosp_numbers(i, self.rates[k])
            # turn to new admissions per day
            admissions[k] = out[k]
            admissions[k] = calc_admission_deltas(admissions[k])
            admissions[k] = rolling_sum(admissions[k], self.los[k])
        return out, admissions


class PennDeath(SIRD):
    """
    PennDeath:
        Make SIR predictions given the number of deaths today
    ----------------------------------------------------------------------
    Parameters:
        N: pop size, Number of people affected
        I: infected, Number of positive tests in region
        R: recovered, number of recoveries
        D: dead, number dead, total
        D_today: dead today, number of people who died today
        hosp_rate: rate of infected admitted to the hospital = 0.15
        icu_rate: rate of infected who need to be in ICU = 0.05
        vent_rate: rate of infected who need ventilators = 0.02
        death_rate: rate of infected who die = 0.01
        contact_reduction: percent contact reduced by social distancing = 0
        t_death: time from infection to death = 20
        t_double: time to double number of infected = 2.
        beta_decay: decay rate of beta, which represents how often a contact results in a new infection = 0
        vent_los: time one patient takes up a ventilator=10
        hos_los: time one patient takes up a normal hospital bed = 7
        icu_los: time one patient takes up an ICU bed = 9
        recover_time: time to get better, to shift from I to R = 23
        birth_rate: Daily rate of birth. Assumed to be 0, as babies are
        hopefully not susceptible

    Attributes:
        rates: dict = rate of hospitalization, icu, and ventilators
        los: dict = length of stay at hospital, icu, and ventilator
        contact_reduction: float = contact reduction
        t_double: float = number of days to double
        intrinsic_growth: float = growth rate
        recover_time: float = time to recover from illness
        beta: float =  how often a contact results in a new infection
        gamma:  float = rate at which an infected person recovers
        beta_decay:  float = decay rate of beta
        r_naught:  float = speadability of disease
        r_t:  float = r_naught after distancing
        S: int = number of susceptible people
        I: int = number of actually infected people
        R: int = number of recovered people
        D: int = number of dead


    Methods:
        sir(n_days):
            run simulation, see docstring for sir
    """
    def __init__(self, N: int, I: int, R: int, D: int, D_today: int,
                 hosp_rate: float=0.15, icu_rate: float=0.05,
                 vent_rate: float=0.02, death_rate: float=0.01,
                 contact_reduction: float=0., t_death: float=20,
                 t_double: float=3.31, beta_decay: float=0, vent_los: float=10,
                 hos_los: float=7, icu_los: float=9, recover_time: float=23,
                 birth_rate: float=0):
        self.rates = {'hospital': hosp_rate,'icu': icu_rate, 'ventilator':vent_rate}
        self.los = dict(zip(self.rates.keys(), [hos_los, icu_los, vent_los]))
        self.contact_reduction = contact_reduction
        self.t_double = t_double
        self.intrinsic_growth = 2**(1/t_double) - 1
        self.recover_time = recover_time
        S = N - (I + R + D)
        gamma = 1/self.recover_time
        beta = ((self.intrinsic_growth + gamma) / S)  * (1-contact_reduction)
        self.r_t = beta/gamma * S
        self.r_naught = self.r_t / (1-contact_reduction)
        if D_today == 0:
            self.I = I
        else:
            infect_t_death = D_today/death_rate
            penn = Penn(S, I, R, 1,
                 hosp_rate, icu_rate ,
                 vent_rate , contact_reduction,
                 t_double, beta_decay, vent_los,
                 hos_los, icu_los, recover_time)
            self.I = penn.sir(t_death)[0]['infected'][-1]
        super().__init__(N, self.I, R, D,beta, gamma, death_rate, birth_rate, beta_decay)


    def sir(self, n_days: int) -> (dict, dict):
        """
        sir: fit SIR model to the data
        -----------------------------------
        Inputs:
            n_days: int = number of days ahead to predict
        Outputs:
            curve: dict :
                susceptible: np.ndarray = history of susceptible pop
                infected: np.ndarray = history of infection
                recovered: np.ndarray = history of recovery
                dead: np.ndarray = history of death
                hospital: np.ndarray = history of hospitalizations
                icu: np.ndarray = history of icu
                ventilator: np.ndarray = history of ventilator
            occupancy: dict of occupancy:
                hospitaliz
                icu
                ventilator
        """
        s, i, r, d = super().sir(n_days)
        out = {}
        admissions = {}
        out['infected'] = i
        out['recovered'] = r
        out['susceptible'] = s
        out['dead'] = d
        for k in self.rates.keys():
            # calculate raw numbers
            out[k] = calc_hosp_numbers(i, self.rates[k])
            # turn to new admissions per day
            admissions[k] = out[k]
            admissions[k] = calc_admission_deltas(admissions[k])
            admissions[k] = rolling_sum(admissions[k], self.los[k])
        return out, admissions

