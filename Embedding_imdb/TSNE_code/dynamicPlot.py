import matplotlib.pyplot as plt
from IPython import display

class KLDynamicPlot:

    def dynamically_plot_kl_divergence(self, kl_divergence, adaptive_learning_rate, 
                                       lower_bound=None, lower_bound_label='', figsize=(7, 4)):
        
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
        # Update the image every 50 steps
        # if i % 50 == 0:
        #     self.dynamically_plot_kl_divergence(figsize=figsize)

        # Update the progress bar at each iteration
        progress_value = (i+1) / total_steps
        self.draw_custom_progress_bar(i, progress_value)

    def draw_custom_progress_bar(self, i, progress_value, bar_length=50):
        # Your custom progress bar implementation using ax.text
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