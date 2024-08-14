import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma


def train_histograms_with_distributions(K, scores_train, model_finals, mixture_distribution='normal'):
    if mixture_distribution not in ['normal', 'gamma']:
        raise ValueError("Invalid mixture_distribution value. Choose either 'normal' or 'gamma'.")

    # Define a color palette with brighter and darker shades
    bright_colors = ['skyblue', 'lightgreen', 'salmon', 'orchid', 'lightcoral', 'lightpink', 'lightgoldenrodyellow', 'lightseagreen']
    dark_colors = ['darkblue', 'darkgreen', 'darkred', 'darkviolet', 'navy', 'forestgreen', 'firebrick', 'teal']

    # Create figure with K subplots
    fig, axs = plt.subplots(K, K, figsize=(14, 12))

    # Define other plot aesthetics
    hist_alpha = 0.7  # Slightly more transparent
    pdf_linewidth = 2
    font_size = 12

    # Plot histograms and distributions
    for i in range(K):
        for j in range(K):
            # Select subplot
            ax = axs[i, j]
            score_data = scores_train[i, j]

            # Plot histogram
            ax.hist(score_data, bins=50, color=bright_colors[i], alpha=hist_alpha, density=True, edgecolor='black')

            if mixture_distribution == 'normal':
                # Extract parameters for normal distributions
                means = [model_finals.distributions[i].means[j] for i in range(K) for j in range(K)]
                covariances = [model_finals.distributions[i].covs[j] for i in range(K) for j in range(K)]

                # Parameters for the current subplot
                mean = means[i * K + j]
                covariance = covariances[i * K + j]

                # Plot normal distribution
                x = np.linspace(score_data.min(), score_data.max(), 100)
                pdf = norm.pdf(x, loc=mean, scale=np.sqrt(covariance))
                ax.plot(x, pdf, color=dark_colors[i], linewidth=pdf_linewidth)

            elif mixture_distribution == 'gamma':
                # Extract parameters for gamma distributions
                shapes = [model_finals.distributions[i].shapes[j] for i in range(K) for j in range(K)]
                rates = [model_finals.distributions[i].rates[j] for i in range(K) for j in range(K)]

                # Parameters for the current subplot
                shape = shapes[i * K + j]
                rate = rates[i * K + j]

                # Plot gamma distribution
                x = np.linspace(score_data.min(), score_data.max(), 100)
                pdf = gamma.pdf(x, a=shape, scale=1 / rate)
                ax.plot(x, pdf, color=dark_colors[i], linewidth=pdf_linewidth)

            # Titles and labels
            ax.set_title(f'OCC {j + 1} on Type {i + 1} Inliers', fontsize=font_size)
            ax.set_xlabel('Score', fontsize=font_size)
            ax.set_ylabel('Density', fontsize=font_size)

            # Grid and ticks
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.tick_params(axis='both', which='major', labelsize=font_size)

            # Set axis limits
            ax.set_xlim(score_data.min(), score_data.max())

    # Add a main title
    fig.suptitle(
        f'Histograms and {mixture_distribution.capitalize()} Distribution Fits for Training Scores Across Different Models',
        fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])  # Adjust to make room for the main title
    plt.show()


def unlabeled_histograms_with_distributions(K, scores_unlabeled, Y_unlabeled, model_finals, mixture_distribution='normal'):
    if mixture_distribution not in ['normal', 'gamma']:
        raise ValueError("Invalid mixture_distribution value. Choose either 'normal' or 'gamma'.")

    # Create a DataFrame from the data
    df = pd.DataFrame({
        'Score': scores_unlabeled[0],
        'Label': Y_unlabeled
    })

    # Define a color palette with brighter and darker shades
    bright_colors = ['skyblue', 'lightgreen', 'salmon', 'orchid', 'lightcoral', 'lightpink', 'lightgoldenrodyellow', 'lightseagreen']
    dark_colors = ['darkblue', 'darkgreen', 'darkred', 'darkviolet', 'navy', 'forestgreen', 'firebrick', 'teal']

    # Create the figure
    plt.figure(figsize=(12, 8))

    # Define order of labels as they appear in the DataFrame
    label_order = pd.Series(df['Label']).unique()

    # Create a mapping from labels to colors for histograms
    label_color_map = dict(zip(label_order, bright_colors))

    # Plot histogram with seaborn
    sns.histplot(df, x='Score', hue='Label', bins=70, palette=label_color_map, alpha=0.7, stat='density')

    # Plot distribution curves
    x = np.linspace(df['Score'].min(), df['Score'].max(), 100)

    if mixture_distribution == 'normal':
        # Extract parameters for normal distributions
        means = [model_finals.distributions[i].means[0] for i in range(K+1)]
        covariances = [model_finals.distributions[i].covs[0] for i in range(K+1)]
        priors = np.array(model_finals.priors)

        # Plot normal distributions with corresponding colors
        for i, label in enumerate(label_order):
            mean = means[i]
            covariance = covariances[i]
            pdf = norm.pdf(x, loc=mean, scale=np.sqrt(covariance))
            plt.plot(x, pdf * priors[i], color=dark_colors[i], linewidth=2, linestyle='--',
                     label=f'Normal Distribution for Label {label}')

        plt.title('Histogram and PDFs of Unlabeled Scores Obtained by OCC 1 (Normal Fitting)', fontsize=14)

    elif mixture_distribution == 'gamma':
        # Extract parameters for gamma distributions
        shapes = [model_finals.distributions[i].shapes[0] for i in range(K+1)]
        rates = [model_finals.distributions[i].rates[0] for i in range(K+1)]
        priors = np.array(model_finals.priors)

        # Plot gamma distributions with corresponding colors
        for i, label in enumerate(label_order):
            shape = shapes[i]
            rate = rates[i]
            pdf = gamma.pdf(x, a=shape, scale=1 / rate)
            plt.plot(x, pdf * priors[i], color=dark_colors[i], linewidth=2, linestyle='--',
                     label=f'Gamma Distribution for Label {label}')

        plt.title('Histogram and PDFs of Unlabeled Scores Obtained by OCC 1 (Gamma Fitting)', fontsize=14)

    # Titles and labels
    plt.xlabel('Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)

    # Grid and legend
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()
