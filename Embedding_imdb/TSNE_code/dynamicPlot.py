import matplotlib.pyplot as plt
from IPython import display

class KLDynamicPlot:
    """
    A class for dynamically plotting KL divergence during gradient descent.

    Methods
    -------
    dynamically_plot_kl_divergence(kl_divergence, adaptive_learning_rate, lower_bound=None, lower_bound_label='', figsize=(7, 4))
        Dynamically plot KL divergence.

    update_image_and_progress_bar(i, total_steps, figsize=(7, 4))
        Update the progress bar and image at each iteration.

    draw_custom_progress_bar(i, progress_value, bar_length=50)
        Draw a custom progress bar.

    Attributes
    ----------
    kl_divergence : list
        List to store KL divergence values.

    Examples
    --------
    >>> from your_module import KLDynamicPlot
    >>> import numpy as np
    >>> kl_plotter = KLDynamicPlot()
    >>> kl_divergence = np.random.rand(100)  # Replace with your actual KL divergence values
    >>> kl_plotter.dynamically_plot_kl_divergence(kl_divergence, adaptive_learning_rate=True)
    >>> kl_plotter.update_image_and_progress_bar(10, total_steps=100)
    >>> kl_plotter.draw_custom_progress_bar(20, progress_value=0.2)
    """

    def dynamically_plot_kl_divergence(self, kl_divergence, adaptive_learning_rate, 
                                       lower_bound=None, lower_bound_label='', figsize=(7, 4)):
        """
        Dynamically plot KL divergence during gradient descent.

        Parameters
        ----------
        kl_divergence : list or numpy.ndarray
            List of KL divergence values.
        adaptive_learning_rate : bool
            Flag indicating whether adaptive learning rate is used.
        lower_bound : float, optional
            Lower bound for the plot, by default None
        lower_bound_label : str, optional
            Label for the lower bound in the legend if one is provided, by default ''
        figsize : tuple, optional
            Size of the figure, by default (7, 4).
        """
        
        fig, ax = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot KL divergence on the upper subplot
        label = 'Custom' if not adaptive_learning_rate else 'Custom + adaptive lr'
        ax[0].plot(kl_divergence, label=label, c='black')        
        ax[0].grid('on')
        ax[0].legend()
        ax[0].set_title("KL divergence through gradient descent")

        # Lower subplot for the progress bar
        ax[1].axis('off')  # Turn off axis for the progress bar subplot

        display.display(fig)
        display.clear_output(wait=True)
        plt.close()

    def update_image_and_progress_bar(self, i, total_steps, figsize=(7, 4)):
        """
        Update the progress bar and image at each iteration.

        Parameters
        ----------
        i : int
            Current iteration step.
        total_steps : int
            Total number of steps.
        figsize : tuple, optional
            Size of the figure, by default (7, 4).
        """
        
        # Update the progress bar at each iteration
        progress_value = (i+1) / total_steps
        self.draw_custom_progress_bar(i, progress_value)

    def draw_custom_progress_bar(self, i, progress_value, bar_length=50):
        """
        Draw a custom progress bar.

        Parameters
        ----------
        i : int
            Current iteration step.
        progress_value : float
            Value indicating the progress (between 0 and 1).
        bar_length : int, optional
            Length of the progress bar, by default 50.
        """
        
        fig, ax = plt.subplots(2, 1, figsize=(7, 4), gridspec_kw={'height_ratios': [3, 1]})

        # Dummy data for the upper subplot
        if i % 50 == 0:
            label = 'Custom'
            ax[0].plot(self.kl_divergence, label=label, c='black')        
            ax[0].grid('on')
            ax[0].legend()
            ax[0].set_title("KL divergence through gradient descent")

        # Progress bar in the lower subplot
        ax[1].axis('off')
        bar = "[" + "#" * int(bar_length * progress_value) + "-" * (bar_length - int(bar_length * progress_value)) + "]"
        ax[1].text(0.5, 0.5, f"Progress: {progress_value * 100:.2f}% {bar}", ha='center', va='center', fontsize=12)

        display.display(fig)
        display.clear_output(wait=True)
        plt.close()