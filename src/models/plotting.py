import matplotlib.pyplot as plt

def plot_and_save_posteriors(posteriors, figures_path):
    """
    Plot posterior samples/traces and save figure in figures_path.

    Arguments:
        posteriors (list) - List of tf.Variable containing posterior samples
        figures_path (str) - Path to figures directory where plots are saved
    """
    n = tf.shape(posteriors)[-1]
    plt.figure(figsize=(12,8))
    for i in range(n):
        #Posterior histogram
        plt.subplot(n, 2, 2*i)
        plt.title("Posterior of " + posteriors[i]['param'])
        plt.hist(posteriors[i]['samples'], color='red', bins=30, density=True, alpha=0.6)
        plt.grid(alpha=0.5)
        #Posterior trace
        plt.subplot(n, 2, 2*i+1)
        plt.title("Trace of " + posteriors[i]['param'])
        plt.plot(posteriors[i]['samples'], color='blue', alpha=0.6)
        plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_path, dpi=500)