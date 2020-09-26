import pymc3 as pm
import arviz as az
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Gather and transform data
data_path = '/home/gerardo/Desktop/Projects/PGA-Analysis/data/driving-data/driving-data.csv'
raw_data = pd.read_csv(data_path)
data = raw_data['average_driving_distance'].to_numpy()

#Declare model
with pm.Model() as model:
    #Switchpoint
    tau = pm.DiscreteUniform("tau", lower=0, upper=len(data)-1)
    #Prior when t <= tau
    mu_1 = pm.Normal("mu_1", mu=280, sd=20)
    sd_1 = pm.HalfNormal("sd_1", sigma=40)
    #Prior when t > tau
    mu_2 = pm.Normal("mu_2", mu=280, sd=20)
    sd_2 = pm.HalfNormal("sd_2", sigma=40)
    #Observations
    idx = np.arange(len(data))
    mu_t = pm.math.switch(tau > idx, mu_1, mu_2)
    sd_t = pm.math.switch(tau > idx, sd_1, sd_2)
    observations = pm.Normal("observations", mu=mu_t, sd=sd_t, observed=data)

#Perform inference
with model:
    step = pm.NUTS()
    trace = pm.sample(50000, tune=5000, step=step)

#Save summary
summary = pm.summary(trace)
with open('sp.summary', 'w') as f:
    summary.to_string(f)

#Plot and save posterior traces
pm.save_trace(trace, 'switchpoint.trace', overwrite=True)
az.plot_trace(trace)
plt.savefig('/home/gerardo/Desktop/Projects/PGA-Analysis/reports/figures/driving-distance-pymc3-posteriors.png')
plt.show()