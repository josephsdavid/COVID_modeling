# COMODELS

Models for COVID - 19

For documentation, see the docstrings!
Much more to come. For help:

```python
import comodels
help(comodels)
help(comodels.PennDeath)
help(comodels.Penn)
```

## Penn Death model

```python
# import the penn model
import matplotlib.pyplot as plt
from comodels import PennDeath

help(PennDeath)
tx = PennDeath(N = 28304596, I = 223, R = 0, D = 3, D_today = 2)

help(PennDeath.sir)

def plot_penn(Pdp: PennDeath, n_days: int) -> None:
    # predict the coming storm and plot it
    curve, admissions = Pdp.sir(n_days)
    fig, ax = plt.subplots(1,3, figsize=(15,5))
    for k, v in curve.items():
        if k not in Pdp.rates.keys() :
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
print(curve.keys())
print(occupancy.keys())
```
