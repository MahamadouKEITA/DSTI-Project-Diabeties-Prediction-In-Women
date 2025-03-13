import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch
import pandas as pd

def plot_diabetic_distribution(df):
    """
    Plots the distribution of diabetic status in the given DataFrame.

    Parameters:
    df (DataFrame): The DataFrame containing the diabetic data.

    Returns:
    None
    """
    # Let's drop the PatientID columns
    df.drop(columns="PatientID", inplace=True)

    # Calculate percentages
    counts = df['Diabetic'].value_counts(normalize=True) * 100

    # Define colors for the binary values
    colors = {0: 'blue', 1: 'red'}

    # Plot histogram
    plt.figure(figsize=(8, 6))
    bars = plt.bar(counts.index, counts, color=[colors[val] for val in counts.index])

    # Add percentage text on top of bars
    for bar, percentage in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 2,
                 f'{percentage:.1f}%', ha='center', va='bottom', fontsize=12, color='black')

    # Customize the plot
    plt.xticks([0, 1], ['Non-Diabetic (0)', 'Diabetic (1)'])
    plt.xlabel("Diabetic Status", fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.title("Distribution of Diabetic Status", fontsize=14)
    plt.ylim(0, 100)  # Ensure the y-axis goes up to 100%

    plt.show()





def plot_boxplot_with_outliers(df, id_var='Diabetic'):
    """
    Plots a boxplot of variables with outliers colored by a specified identifier variable.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    id_var (str): The column name to use for coloring outliers (default is 'Diabetic').

    Returns:
    None
    """
    # Melt the DataFrame to long-form for easier plotting
    df_melted = df.melt(id_vars=id_var, var_name='Variable', value_name='Value')

    # Initialize the plot
    plt.figure(figsize=(10, 7))
    sns.set(style="whitegrid")

    # Generate a boxplot without the outliers
    sns.boxplot(
        x='Variable',
        y='Value',
        data=df_melted,
        width=0.6,
        palette='Set2',  # Unique colors per variable
        showfliers=False  # Hide default outliers
    )

    # Overlay individual points and color outliers based on id_var status
    for i, variable in enumerate(df_melted['Variable'].unique()):
        subset = df_melted[df_melted['Variable'] == variable]

        # Calculate whisker limits for outliers
        Q1 = np.percentile(subset['Value'], 25)
        Q3 = np.percentile(subset['Value'], 75)
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR

        # Identify outliers
        outliers = subset[(subset['Value'] < lower_limit) | (subset['Value'] > upper_limit)]

        # Plot outliers as scatter points with colors based on id_var
        plt.scatter(
            [i] * len(outliers),  # X-coordinates (aligned with boxplot position)
            outliers['Value'],  # Y-coordinates
            c=outliers[id_var].apply(lambda x: 'blue' if x == 0 else 'red'),
            edgecolor='black',
            s=50,  # Marker size
            zorder=5,  # Ensure points are on top
            label='_nolegend_'
        )

    # Add legend for color meaning
    legend_handles = [
        Patch(color='blue', label=f'Non-{id_var} (0)'),
        Patch(color='red', label=f'{id_var} (1)')
    ]
    plt.legend(handles=legend_handles, loc='upper right', title=id_var)

    # Rotate x-axis labels
    plt.xticks(rotation=45, fontsize=10)

    # Customize the plot
    plt.title(f"Boxplot of Variables with Outliers Colored by {id_var} Status", fontsize=14)
    plt.xlabel("Variables", fontsize=12)
    plt.ylabel("Values", fontsize=12)

    # Adjust layout for clarity
    plt.tight_layout(pad=2)

    # Show the plot
    plt.show()




def plot_histograms_with_outliers(df, features, colors, title="Histograms of Features with Outliers"):
    """
    Plots histograms of specified features with unique colors.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    features (list): List of feature names to plot.
    colors (list): List of colors for each feature.
    title (str): The title of the plot.

    Returns:
    None
    """

    # Set Seaborn style for nice visuals
    sns.set_theme(style="whitegrid", palette="muted", color_codes=True)

    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Adjust the figure size as needed
    fig.suptitle(title, fontsize=16)

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Plot histograms for each feature with unique colors
    for i, (feature, color) in enumerate(zip(features, colors)):
        sns.histplot(df[feature].dropna(), bins=100, kde=True, ax=axes[i], color=color)
        axes[i].set_title(f"Distribution of {feature}", fontsize=12)
        axes[i].set_xlabel(feature, fontsize=10)
        axes[i].set_ylabel("Frequency", fontsize=10)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Ensure the main title does not overlap
    plt.show()



def plot_outlier_proportions(df, features, id_var='Diabetic', title="Proportion of Outliers by Diabetic Status"):
    """
    Plots the proportion of outliers for specified features by a given identifier variable.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    features (list): List of feature names to analyze.
    id_var (str): The column name to use for grouping (default is 'Diabetic').
    title (str): The title of the plot.

    Returns:
    None
    """
    # Set Seaborn style for nice visuals
    sns.set_theme(style="whitegrid", palette="muted", color_codes=True)

    # Define a function to detect outliers using IQR method
    def detect_outliers_iqr(data):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (data < lower_bound) | (data > upper_bound)

    # Create a DataFrame to store outlier-related data
    df_outliers = pd.DataFrame(index=df.index)

    # Proportion of Outliers per Category (id_var=0 and id_var=1)
    outlier_proportions = []

    # Detect and save outliers in df_outliers
    for feature in features:
        # Detect outliers for the current feature
        is_outlier = detect_outliers_iqr(df[feature])

        # Add the actual outlier values to df_outliers
        outlier_column_name = f"{feature}_outliers"
        df_outliers[outlier_column_name] = is_outlier.astype(int)  # Add binary indicator to df_outliers (1 = outlier, 0 = non-outlier)
        df_outliers[outlier_column_name] = df[feature].where(is_outlier, other=0)

        # Calculate proportion of outliers for each group
        proportions = df.groupby(id_var)[feature].apply(
            lambda x: is_outlier.loc[x.index].mean() * 100
        )
        outlier_proportions.append(proportions)

    # Combine results into a DataFrame for easier plotting
    outlier_summary = pd.DataFrame(outlier_proportions, index=features)
    outlier_summary.columns = [f'Non-{id_var} (%)', f'{id_var} (%)']

    # Plot the proportions as a grouped barplot
    outlier_summary.plot(kind='bar', figsize=(12, 6), color=['blue', 'red'])
    plt.title(title, fontsize=16)
    plt.xlabel("Features", fontsize=12)
    plt.ylabel("Outlier Proportion (%)", fontsize=12)
    plt.legend(title=f"{id_var} Status")
    plt.tight_layout()
    plt.show()




def plot_correlation_heatmap(df, title="Enhanced Correlation Heatmap (Purple)", figsize=(12, 10), cmap_color="purple"):
    """
    Plots an enhanced correlation heatmap with a specified colormap.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    title (str): The title of the plot.
    figsize (tuple): The size of the figure.
    cmap_color (str): The base color for the colormap.

    Returns:
    None
    """
    # Set a larger figure size and clean style for readability
    plt.figure(figsize=figsize)
    sns.set_theme(style="white", font_scale=1.2)

    # Compute the correlation matrix
    correlation_matrix = df.corr()

    # Create a mask to hide the upper triangle (for a clean look)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    # Define a colormap
    cmap = sns.light_palette(cmap_color, as_cmap=True)  # Light gradient colormap

    # Create the heatmap
    sns.heatmap(
        correlation_matrix, 
        mask=mask,  # Apply the mask
        cmap=cmap,  # Use the specified colormap
        annot=True,  # Show correlation coefficients
        fmt=".2f",   # Format numbers to 2 decimal places
        annot_kws={"size": 10},  # Adjust annotation size
        linewidths=0.5,  # Add lines between cells for better readability
        cbar_kws={"shrink": 0.8},  # Adjust color bar size
        square=True  # Keep cells square-shaped
    )

    # Add a title
    plt.title(title, fontsize=16, pad=20)

    # Display the plot
    plt.show()










