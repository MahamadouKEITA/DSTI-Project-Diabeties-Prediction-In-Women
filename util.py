import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch

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




