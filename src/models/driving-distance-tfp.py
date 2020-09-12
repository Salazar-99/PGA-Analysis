import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import pandas as pd

#For convenience
tfd = tfp.distributions
tfb = tfp.bijectors

#Gather and transform data
data_path = '/home/gerardo/Desktop/Projects/PGA-Analysis/data/driving-data/driving-data.csv'
raw_data = pd.read_csv(data_path)
data = tf.constant(data['average_driving_distance'].to_numpy(), dtype=tf.float32)

#TODO: Write docstrings for functions

#Define joint log-probability of switch-point model
def joint_log_prob(data, mu_1, mu_2, tau, sd):
    #Define probabilistic model
    rv_mu_1 = tfd.Normal(loc=250, scale=25)
    rv_mu_2 = tfd.Normal(loc=250, scale=25)
    rv_tau = tfd.Uniform(low=1, high=tf.shape(data)[-1])
    rv_sd = tfd.Normal(loc=25, scale=10)
    mu = tf.gather([mu_1, mu_2], indices=tf.cast(tau*tf.cast(tf.size(data), dtype=tf.float32) <= 
                                         tf.cast(tf.range(tf.size(data)), dtype=tf.float32), dtype=tf.int32))
    rv_observations = tfd.Normal(loc=mu, scale=sd)
    #Return log-likelihood
    return (rv_mu_1.log_prob(mu_1)+
            rv_mu_2.log_prob(mus_2)+
            rv_tau.log_prob(tau)+
            rv_sd.log_prob(sd)+
            tf.reduce_sum(rv_observations.log_prob(data)))

#Closure over joint_log_prob
def unnormalized_log_posterior(mu_1, mu_2, tau, sd):
    return joint_log_prob(data, mu_1, mu_2, tau, sd)

#Define HMC sampler
@tf.function
def sample(n_steps, n_burnin_steps, data, unnormalized_log_posterior):
    #Markov Chain start state (mu_1, mu_2, tau)
    initial_chain_state = [tf.cast(tf.reduce_mean(data), tf.float32) * tf.ones([], dtype=tf.float32),
                           tf.cast(tf.reduce_mean(data), tf.float32) * tf.ones([], dtype=tf.float32),
                           0.5 * tf.ones([], dtype=tf.float32),
                           tf.cast(tf.math.reduce_std(data), tf.float32) * tf.ones([], dtype=tf.float32)]
    #Define bijectors to ensure sampler states remain in support of priors
    unconstraining_bijectors = [tfb.Exp(),
                                tfb.Exp(),
                                tfb.Sigmoid(),
                                tfb.Exp()]
    #Define HMC kernel with adaptive step size and appropriate bijectors
    kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=unnormalized_log_posterior,
                num_leapfrog_steps=2,
                step_size=0.2,
                state_gradients_are_stopped=True),
            bijector=unconstraining_bijectors)
    kernel = tfp.mcmc.SimpleStepSizeAdaptation(inner_kernel=kernel, 
                                               num_adaptation_steps=int(n_burn_in_steps * 0.8))
    #Sample chain
    [mu_1_samples, mu_2_samples, tau_samples, sd_samples], kernel_results = sample_chain(num_results=n_steps, 
                                                                             num_burnin_steps=n_burnin_steps,
                                                                             current_state=initial_chain_state,
                                                                             kernel=kernel)
    #Return posterior samples
    return [mu_1_samples. mu_2_samples, tau_samples, sd_samples]

#TODO: Write inference logic and plot/save posteriors