import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('tkagg')
plt.style.use('seaborn-paper')
from comodels.sir import Penn_detect_prob, rolling_sum

print(Penn_detect_prob.__doc__)

tx = Penn_detect_prob(28304596, 223, 0, 1/8)
tx_good = Penn_detect_prob(28304596, 223, 0, 1/8, contact_reduction = 0.333)

fig, axs = plt.subplots(2,2)
for k, v in tx.sir(180).items():
    if k in tx.rates.keys():
        axs[0,0].plot(v, label=k)
        axs[0,0].legend()
axs[0,0].set_title('Hospital Resource Usage, No social distancing, TX')
for k, v in tx.sir(180).items():
    if k not in tx.rates.keys():
        axs[1,0].plot(v, label=k)
        axs[1,0].legend()
axs[1,0].set_title('SIR chart, No social distancing, TX')
for k, v in tx_good.sir(180).items():
    if k in tx_good.rates.keys():
        axs[0,1].plot(v, label=k)
        axs[0,1].legend()
axs[0,1].set_title('Hospital Resource Usage, Social contact reduced by 33%, TX')
for k, v in tx_good.sir(180).items():
    if k not in tx_good.rates.keys():
        axs[1,1].plot(v, label=k)
        axs[1,1].legend()
axs[1,1].set_title('SIR chart, Social contact reduced by 33%, TX')
plt.suptitle("Penn model of TX, given sensitivity of test = 0.5")
plt.show()

