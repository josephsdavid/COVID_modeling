import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('tkagg')
plt.style.use('seaborn-paper')
from comodels import PennDeath

tx = PennDeath(330460832, 14250, 125, 205, 57, death_rate = 0.01, birth_rate=0, contact_reduction = 0.)

def plot_penn(Pdp: PennDeath, n_days: int) -> None:
    curve, admissions = Pdp.sir(n_days)
    fig, ax = plt.subplots(1,3, figsize=(15,5))
    for k, v in curve.items():
        if k not in tx.rates.keys() :
            ax[0].plot(v, label=k)
            ax[0].legend()
        else:
            ax[1].plot(v, label=k)
            ax[1].legend()
    ax[1].set_title('Hospital Resource Usage')
    ax[0].set_title('SIR curve')
    for k, v in admissions.items():
        ax[2].plot(v, label = k)
        ax[2].legend()
    ax[2].set_title('Additional Resource Usage by day')
    fig.suptitle(f"No social distancing, total deaths = {int(max(curve['dead']))}")
    plt.show()


plot_penn(tx, 120)
