import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch
import pandas as pd
from scipy.stats import mannwhitneyu
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


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



def plot_pairplot(df, hue="Diabetic", title="Pair Plot of Features Colored by Diabetic Status"):
    """
    Plots a pair plot of features colored by a specified hue.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    hue (str): The column name to use for coloring the plots (default is 'Diabetic').
    title (str): The title of the plot.

    Returns:
    None
    """
    # Custom color palette for the binary class: 0 -> Blue, 1 -> Red
    palette = {0: "blue", 1: "red"}

    # Create the pair plot
    g = sns.pairplot(
        df, 
        hue=hue, 
        palette=palette,
        diag_kind="kde",  # KDE plots on the diagonal
        markers=["o", "D"],  # Use distinct markers for each class
        corner=True,
        plot_kws={'alpha': 0.7, 's': 60, 'edgecolor': 'black'}  # Adjust marker size, transparency, and edges
    )

    # Add a main title
    g.figure.suptitle(title, y=1.02, fontsize=16)

    # Display the plot
    plt.show()



def plot_horizontal_bar_in_rows(df, binary_column):
    """
    Creates a row of horizontal bar plots for each numeric column in the DataFrame with a specified binary column as X.

    Args:
    df (pd.DataFrame): Input DataFrame.
    binary_column (str): The binary column to use as the grouping variable.
    """
    # Ensure the binary column is present
    if binary_column not in df.columns:
        raise ValueError(f"The specified column '{binary_column}' is not in the DataFrame.")
    
    # Convert the binary column to a string if needed
    df[binary_column] = df[binary_column].astype(str)

    # Filter numeric columns excluding the binary column
    numeric_columns = [col for col in df.select_dtypes(include='number').columns if col != binary_column]
    
    # Set up the plot grid
    num_vars = len(numeric_columns)
    fig, axes = plt.subplots(num_vars, 1, figsize=(10, 4 * num_vars), constrained_layout=True)
    
    # If there's only one numeric column, axes might not be iterable
    if num_vars == 1:
        axes = [axes]

    for ax, col in zip(axes, numeric_columns):
        # Prepare data for plotting
        grouped_data = df.groupby(binary_column)[col].mean().reset_index()
        
        # Plot the data horizontally
        sns.barplot(
            y=binary_column, 
            x=col, 
            data=grouped_data, 
            ax=ax, 
            palette={"0": "blue", "1": "red"},
            orient="h"  # Horizontal orientation
        )
        
        # Customize the subplot
        ax.set_title(f"Bar Plot of {col} by {binary_column}", fontsize=14)
        ax.set_xlabel(f"Mean {col}", fontsize=12)
        ax.set_ylabel(binary_column, fontsize=12)
        ax.grid(axis="x", linestyle="--", alpha=0.7)

    plt.show()





def mann_whitney_test(df, binary_var, alpha=0.05):
    """
    Separates a DataFrame into two groups based on a binary variable and applies 
    the Mann-Whitney U test to all numeric variables. Includes a significance column.
    
    Parameters:
    - df: pd.DataFrame
        The DataFrame containing the data.
    - binary_var: str
        The name of the binary variable (column) used for grouping.
    - alpha: float, optional (default=0.05)
        The significance level for determining if a result is statistically significant.
        
    Returns:
    - results: pd.DataFrame
        A DataFrame containing the U statistic, p-value, and significance for each numeric variable.
    """
    # Validate input
    if binary_var not in df.columns:
        raise ValueError(f"The binary variable '{binary_var}' is not in the DataFrame.")
    
    # Identify numeric columns
    numeric_vars = df.columns.drop(binary_var)
    if numeric_vars.empty:
        raise ValueError("No numeric variables found in the DataFrame.")

    # Prepare results
    results = []

    # Separate the groups based on the binary variable
    group_0 = df[df[binary_var] == "0"]
    group_1 = df[df[binary_var] == "1"]

    # Ensure both groups have data
    if group_0.empty or group_1.empty:
        raise ValueError("One of the groups is empty. Check the binary variable values.")

    # Apply the Mann-Whitney U test for each numeric variable
    for var in numeric_vars:
        u_stat, p_value = mannwhitneyu(group_0[var], group_1[var], alternative='two-sided')
        is_significant = p_value < alpha
        results.append({
            "Variable": var, 
            "U statistic": u_stat, 
            "p-value": p_value, 
            "is_significant": is_significant
        })

    # Convert results to a DataFrame
    results = pd.DataFrame(results)

    print(f"Results of Mann-Whitney U Test for '{binary_var}':")
    print(results)






def evaluate_and_plot_models(df, target="Diabetic", n_splits=5, random_state=42):
    """
    Evaluate models using Stratified K-Fold cross-validation and plot the results.

    Parameters:
    - df: pd.DataFrame
        The DataFrame containing the data.
    - target: str, optional (default="Diabetic")
        The target variable name.
    - n_splits: int, optional (default=5)
        Number of folds in Stratified K-Fold.
    - random_state: int, optional (default=42)
        Random state for reproducibility.
    - title: str, optional
        Title of the plot.

    Returns:
    None
    """
    # Setup the variables
    y = target
    X = df.columns.drop(y)

    df[y] = df[y].astype(int)  # Correct the type of the target variable

    X = df[X]
    y = df[y]

    # Baseline model (proportion of the dominant class)
    dominant_class_proportion = y.value_counts(normalize=True).max()

    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=100),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    }

    # Perform K-Fold evaluation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = {name: [] for name in models.keys()}

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name].append(accuracy)

    # Plot the results
    plt.figure(figsize=(12, 8))

    # Generate unique colors for each model
    colors = ['lightblue', 'lightgreen', 'purple', 'lightpink', 'lightyellow']

    # Boxplot settings
    boxprops = dict(linewidth=2, color="black")
    whiskerprops = dict(linewidth=2, linestyle="--", color="gray")
    capprops = dict(linewidth=2, color="black")
    medianprops = dict(linewidth=2, color="blue")
    flierprops = dict(marker='o', color='red', alpha=0.6)

    # Create the boxplot
    box = plt.boxplot(
        results.values(),
        labels=results.keys(),
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        medianprops=medianprops,
        flierprops=flierprops,
        patch_artist=True
    )

    # Assign unique colors to each box
    for patch, color in zip(box['boxes'], colors[:len(results)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Plot the baseline
    plt.axhline(y=dominant_class_proportion, color="red", linestyle="--", linewidth=2, label="Baseline")

    # Add titles and labels
    title=f"{n_splits}-Fold Accuracy Comparison"
    
    plt.title(title, fontsize=16, fontweight="bold")
    plt.ylabel("Accuracy", fontsize=14)
    plt.xlabel("Models", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_feature_importance(df, importance_type='gain', max_features=10, random_state=42):
    """
    Train an XGBoost model and plot the feature importance.

    Parameters:
    - df: pd.DataFrame
        DataFrame containing the feature matrix and target variable.
    - importance_type: str, optional (default='gain')
        Type of importance ('weight', 'gain', 'cover', 'total_gain', 'total_cover').
    - max_features: int, optional (default=10)
        Maximum number of features to display.
    - random_state: int, optional (default=42)
        Random state for reproducibility.

    Returns:
    - None
    """
    # Define the target variable and feature matrix
    y = df['Diabetic'].astype(int)
    X = df.drop(columns=['Diabetic'])

    # Train the XGBoost model
    model = XGBClassifier(eval_metric="logloss", random_state=random_state)
    model.fit(X, y)

    # Get feature importance scores
    importance = model.get_booster().get_score(importance_type=importance_type)
    
    # Convert to a DataFrame
    importance_df = pd.DataFrame(
        {'Feature': list(importance.keys()), 'Importance': list(importance.values())}
    ).sort_values(by="Importance", ascending=False)
    
    # Select the top features
    top_features = importance_df.head(max_features)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
    plt.xlabel("Importance", fontsize=14)
    plt.ylabel("Feature", fontsize=14)
    plt.title(f"Top {max_features} Features by {importance_type.capitalize()}", fontsize=16, fontweight="bold")
    plt.gca().invert_yaxis()  # Reverse order for better visualization
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()



def detect_outliers_iqr(data):
    """
    Detect outliers using the IQR method.
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data < lower_bound) | (data > upper_bound)

def handle_outliers(df, outliers_features = ['SerumInsulin', 'TricepsThickness', 'DiabetesPedigree', 'Age']):
    """
    Detects outliers in the specified features of df, creates a DataFrame to store outlier-related data,
    and updates the original DataFrame with new columns for outliers.
    
    Parameters:
    df (DataFrame): The main DataFrame.
    outliers_features (list): List of features to detect outliers in.
    
    Returns:
    DataFrame: The updated DataFrame with new columns for outliers.
    """
    # Create a DataFrame to store outlier-related data
    df_outliers = pd.DataFrame(index=df.index)

    # Detect and save outliers in df_outliers
    for feature in outliers_features:
        # Detect outliers for the current feature
        is_outlier = detect_outliers_iqr(df[feature])
        
        # Add the actual outlier values to df_outliers
        outlier_column_name = f"{feature}_outliers"
        df_outliers[outlier_column_name] = df[feature].where(is_outlier, other=0)
    
    # Ensure proper alignment of indices between df and df_outliers
    df_outliers = df_outliers.reindex(df.index)

    # Iterate through columns in df_outliers
    for col in df_outliers.columns:
        if col.endswith('_outliers'):  # Check if the column name ends with '_outliers'
            # Extract the corresponding base column name
            base_col = col.replace('_outliers', '')
            
            # Check if the base column exists in df
            if base_col in df.columns:
                # Handle NaN values by filling them with 0 or other strategies
                df_outliers[col] = df_outliers[col].fillna(0)  # Replace NaN with 0 in df_outliers
                df[base_col] = df[base_col].fillna(0)          # Replace NaN with 0 in df
                
                # Create a new column in df by multiplying
                df[f"{col}"] = df[base_col] * df_outliers[col]

    return df
























