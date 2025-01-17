import pandas as pd
from sklearn.datasets import load_iris
from src.bivariate_functions import categorize_into_deciles_with_stats, plot_data_by_varname

def main(df, target_column, drop_cols, n_deciles=10, f_decile_tree=False):
    """
    Main function to process a DataFrame, evaluate columns using bivariate statistics,
    and visualize the results.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing data to analyze.
        target_column (str): The target column to evaluate against.
        drop_cols (list): Columns to exclude from the evaluation.
        n_deciles (int): Number of deciles for categorization (default=10).
        f_decile_tree (bool): Flag for alternate decile categorization logic (default=False).

    Returns:
        pd.DataFrame: DataFrame containing bivariate statistics for evaluated columns.
        dict: Dictionary of visualizations for each evaluated column.
    """
    # Validate input columns
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is not in the DataFrame.")
    for col in drop_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' in drop_cols is not in the DataFrame.")

    # Initialize an empty DataFrame to store results
    bivariate_statistics = pd.DataFrame()
    visualizations = {}

    # Identify columns to evaluate
    cols_to_evaluate = list(set(df.columns) - set([target_column]) - set(drop_cols))

    for column in cols_to_evaluate:
        print(f"Processing column: {column}")

        # Categorize and compute statistics
        try:
            result = categorize_into_deciles_with_stats(
                df, column, target_column, n_deciles=n_deciles, f_decile_tree=f_decile_tree
            )
            bivariate_statistics = pd.concat([bivariate_statistics, result], ignore_index=True)
        except Exception as e:
            print(f"Error processing column '{column}': {e}")
            continue

    # Generate visualizations
    for var in cols_to_evaluate:
        try:
            visualizations[var] = plot_data_by_varname(bivariate_statistics, var, target_column)
        except Exception as e:
            print(f"Error generating plot for '{var}': {e}")

    return bivariate_statistics, visualizations

if __name__ == "__main__":
    # Load the iris dataset
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

    # Set the target column and columns to drop
    target_column = "sepal length (cm)"
    drop_cols = []

    # Run the main analysis
    try:
        results, visualizations = main(df, target_column, drop_cols)
        print("Bivariate analysis completed.")
        print(results)
        for graph_name, graph in visualizations.keys():
            print(f"Visualization for {graph_name}: {graph}")
    except Exception as e:
        print(f"An error occurred: {e}")
