import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('tkagg')
plt.style.use('seaborn-paper')
from comodels import Penn

print(Penn.__doc__)

print(Penn.sir.__doc__)

tx = Penn(28304596, 223, 0, 1/8)
tx_good = Penn(28304596, 223, 0, 1/8, contact_reduction = 0.33)

fig, axs = plt.subplots(3,2)
for k, v in tx.sir(180)[0].items():
    if k in tx.rates.keys():
        axs[1,0].plot(v, label=k)
        axs[1,0].legend()
axs[1,0].set_title('Hospital Resource Usage, No social distancing, TX')
for k, v in tx.sir(180)[0].items():
    if k not in tx.rates.keys():
        axs[0,0].plot(v, label=k)
        axs[0,0].legend()
axs[0,0].set_title('SIR chart, No social distancing, TX')
for k, v in tx.sir(180)[1].items():
        axs[2,0].plot(v, label=k)
        axs[2,0].legend()
axs[2,0].set_title('Daily Hospital admissions, no social distancing, TX')
for k, v in tx_good.sir(180)[0].items():
    if k in tx_good.rates.keys():
        axs[1,1].plot(v, label=k)
        axs[1,1].legend()
axs[1,1].set_title('Hospital Resource Usage, Social contact reduced by 33%, TX')
for k, v in tx_good.sir(180)[0].items():
    if k not in tx_good.rates.keys():
        axs[0,1].plot(v, label=k)
        axs[0,1].legend()
axs[0,1].set_title('SIR chart, Social contact reduced by 33%, TX')
for k, v in tx_good.sir(180)[1].items():
        axs[2,1].plot(v, label=k)
        axs[2,1].legend()
axs[2,1].set_title('Daily Hospital admissions, Social Contact reduced by 33%, TX')
plt.suptitle("Penn model of TX, given sensitivity of test = 1/8")
plt.show()

