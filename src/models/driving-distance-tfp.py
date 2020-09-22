import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from plotting import plot_and_save_posteriors

#For convenience
tfd = tfp.distributions
tfb = tfp.bijectors

#Argument parser 
parser = argparse.ArgumentParser(description='Specify sampler parameters')
parser.add_argument('--n_steps', type=int, help="Number of sampler steps")
parser.add_argument('--n_burnin_steps', type=int, help="Number of steps to discard from initial part of sampler chain")
args = parser.parse_args()

#Gather and transform data
data_path = '/home/gerardo/Desktop/Projects/PGA-Analysis/data/driving-data/driving-data.csv'
raw_data = pd.read_csv(data_path)
data = tf.constant(raw_data['average_driving_distance'].to_numpy(), dtype=tf.float32)

def joint_log_prob(data, mu_1, mu_2, tau, sd):
    """
    Calculate and return log-posterior. This function effectively defines the model.

    Arguments:
        data (tf.Constant) - 1D Tensor containing data
        mu_1 (tf.Variable) - Mean of first part of distribution
        mu_2 (tf.Variable) - Mean of second part of distribution
        tau (tf.Variable) - Switch-point of model
        sd (tf.Variable) - Standard deviation of parent Gaussian

    Returns:
        log_posterior (tf.Variable) - Log posterior of model
    """
    #Define probabilistic model
    rv_mu_1 = tfd.Normal(loc=250., scale=25.)
    rv_mu_2 = tfd.Normal(loc=250., scale=25.)
    rv_tau = tfd.Uniform(low=1, high=tf.shape(data)[-1])
    rv_sd = tfd.Normal(loc=25., scale=10.)
    mu = tf.gather([mu_1, mu_2], indices=tf.cast(tau*tf.cast(tf.size(data), dtype=tf.float32) <= 
                                         tf.cast(tf.range(tf.size(data)), dtype=tf.float32), dtype=tf.int32))
    rv_observations = tfd.Normal(loc=mu, scale=sd)
    log_posterior = (rv_mu_1.log_prob(mu_1)+
                    rv_mu_2.log_prob(mu_2)+
                    rv_tau.log_prob(tau)+
                    rv_sd.log_prob(sd)+
                    tf.reduce_sum(rv_observations.log_prob(data)))
    return log_posterior

def unnormalized_log_posterior(mu_1, mu_2, tau, sd):
    """
    Closure over joint_log_prob for use in sampler.
    Args equivalent to those in joint_log_prob
    """
    return joint_log_prob(data, mu_1, mu_2, tau, sd)

@tf.function
def sample(n_steps, n_burnin_steps, data, unnormalized_log_posterior):
    """
    Run HMC sampler and return samples of posterior for each parameter

    Arguments:
        n_steps (int) - Total number of steps the sampler will take
        n_burnin_steps (int) - Number of steps at the beginning of the chain to be discarded
        data (tf.Constant) - Tensor containing data
        unnormalized_log_posterior (function) - Closure over log_posterior

    Returns:
        posteriors (list) - List containing dictionaries with posterior samples and names for each parameter
    """
    #Markov Chain start state (mu_1, mu_2, tau, sd)
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
                                               num_adaptation_steps=int(n_burnin_steps * 0.8))
    #Sample chain
    [mu_1_samples, mu_2_samples, tau_samples, sd_samples], kernel_results = tfp.mcmc.sample_chain(num_results=n_steps, 
                                                                                    num_burnin_steps=n_burnin_steps,
                                                                                    current_state=initial_chain_state,
                                                                                    kernel=kernel)
    #Return posterior samples
    posteriors = [{"param": "mu_1", "samples": mu_1_samples},
                  {"param": "mu_2", "samples": mu_2_samples},
                  {"param": "tau", "samples": tau_samples},
                  {"param": "sd", "samples": sd_samples}]
    return posteriors

#Run sampler
posteriors = sample(n_steps=args.n_steps, 
                    n_burnin_steps=args.n_burnin_steps, 
                    data=data, 
                    unnormalized_log_posterior=unnormalized_log_posterior)

#Plot/save posteriors
figures_path = '/home/gerardo/Desktop/Projects/PGA-Analysis/reports/figures/driving-distance-tfp.png'
plot_and_save_posteriors(posteriors, figures_path)

