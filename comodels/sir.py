import numpy as np

class SIR:
    def __init__(self, S: int, I: int, R: int, beta: float,
                 gamma: float, beta_decay: float=0) -> None:
        if beta_decay < 0 or beta_decay > 1:
            raise ValueError('bad beta decay')
        self.S = S
        self.I = I
        self.R = R
        self.N = sum([S,I,R])
        self.beta = beta
        self.gamma = gamma
        self.beta_decay = beta_decay
        self.s = [S]
        self.i = [I]
        self.r = [R]

    def _rectify(self, x: float) -> float:
        out = 0 if x <= 0 else x
        return out

    def _Sn(self, S: int, I:int) -> float:
        return self._rectify((-self.beta * S * I) + S)

    def _In(self, S: int, I:int) -> float:
        return self._rectify((self.beta * S * I - self.gamma * I) + I)

    def _Rn(self, I: int, R: int) -> float:
        return self._rectify((self.gamma * I + R))

    def _step(self, S: int, I: int, R: int) -> (float, float, float):
        Sn = self._Sn(S, I)
        Rn = self._Rn(I, R)
        In = self._In(S, I)
        scale = self.N / (Sn + Rn + In)
        S = Sn * scale
        I = In * scale
        R = Rn * scale
        return S, I , R

    def sir(self, n_days: int) -> (np.ndarray, np.ndarray, np.ndarray):
        S = self.S
        I = self.I
        R = self.R
        self.s = [S]
        self.i = [I]
        self.r = [R]
        for day in range(n_days):
            S, I, R = self._step(S, I, R)
            self.beta *= (1-self.beta_decay)
            self.s.append(S)
            self.i.append(I)
            self.r.append(R)
        return (np.asarray(x) for x in [self.s, self.i, self.r])


class SIRD(SIR):
    def __init__(self, N: int, I: int, R: int, D: int, beta: float, gamma: float, mu: float,
                 birth_rate: float, beta_decay: float) -> None:
        S = N - (I+R+D)
        super().__init__(S,I,R,beta,gamma, beta_decay)
        self.N = N
        self.birth_rate = birth_rate
        self.D = D
        self.mu = mu
        self.d = [D]

    def _Sn(self, S: int, I: int) -> float:
        out = (self.birth_rate - self.mu * S - self.beta * S * I) + S
        return self._rectify(out)

    def _In(self, S: int, I: int) -> float:
        lhs = self.beta * S * I
        rhs = self.gamma * I - self.mu * I
        out = lhs - rhs + I
        return self._rectify(out)

    def _Rn(self, I: int, R: int) -> float:
        lhs = self.gamma * I
        rhs = self.mu * R
        out = lhs - rhs + R
        return self._rectify(out)

    def _step(self, S: int, I: int, R: int) -> (float, float, float, float):
        S, I, R = super()._step(S, I, R)
        return S, I, R

    def sir(self, n_days: int) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        S = self.S
        I = self.I
        R = self.R
        D = self.D
        self.s = [S]
        self.i = [I]
        self.r = [R]
        self.d = [D]
        for day in range(n_days):
            S, I, R = self._step(S, I, R)
            self.d.append(I*self.mu)
            self.beta *= (1-self.beta_decay)
            self.s.append(S)
            self.i.append(I)
            self.r.append(R)
        return (np.asarray(x) for x in [self.s, self.i, self.r, self.d])




