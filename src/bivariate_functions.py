import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML
from matplotlib.table import Table



def jitter(a_series, noise_reduction=1000000):
    return (np.random.random(len(a_series))*a_series.std()/noise_reduction)-(a_series.std()/(2*noise_reduction))
    
def numeric_decilecuts(df, column_name, n_deciles=10):
    # Use qcut on non-null values to create deciles
    df['X_decile'] = pd.qcut(df[column_name].dropna()+ jitter(df[column_name].dropna()),
                             n_deciles,
                             labels=range(1, n_deciles + 1),
                             duplicates='drop')

    # Only add a 'Missing' category if there are NaN values
    if df[column_name].isna().any():
        df['X_decile'] = df['X_decile'].cat.add_categories(['Missing'])
        df['X_decile'] = df['X_decile'].fillna('Missing')   

    return df
def numeric_treecuts(df, column_name, Y):

    breakpoints = tree_cuts(df[df[column_name].notnull()],column_name, Y)
    # Apply the updated function to the 'X_max' column with the breakpoints DataFrame
    df['X_decile'] = df[column_name].apply(lambda x: find_group(x, breakpoints))
    # Only add a 'Missing' category if there are NaN values
    df.loc[df[column_name].isna(), "X_decile"]= "Missing"
    return df

def tree_cuts(df, X_var, Y_var): 
    X = df[[X_var]]  # Features
    y = df[Y_var]  # Target

    # Initialize and train the decision tree
    tree = DecisionTreeRegressor(random_state=42, min_samples_leaf=int(len(X)/10))
    tree.fit(X, y)

    # Extract information from the tree
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    values = tree.tree_.value
    impurities = tree.tree_.impurity
    samples = tree.tree_.n_node_samples

    # Initialize the table
    table = pd.DataFrame(columns=["min", "max", "sample", "value", "squared error"])

    # Function to traverse the tree and extract information for the table
    def recurse(node, lower_bound=float('-inf'), upper_bound=float('inf')):
        if children_left[node] == children_right[node]:  # It's a leaf
            value = values[node][0, 0]
            sample_count = samples[node]
            mse = impurities[node] * samples[node]  # Total squared error for this node
            table.loc[len(table)] = [lower_bound, upper_bound, sample_count, value, mse]
        else:
            # Continue the recursion on both children
            if children_left[node] != -1:
                recurse(children_left[node], lower_bound, threshold[node])
            if children_right[node] != -1:
                recurse(children_right[node], threshold[node], upper_bound)

    # Start recursion from root
    recurse(0)
    return table

def find_group(value, breakpoints):
    """
    Determines the group index for a given value based on specified breakpoints.

    Parameters:
    - value: The value to classify into a group.
    - breakpoints: DataFrame with 'min' and 'max' columns defining group boundaries.

    Returns:
    - The index of the group if a match is found, otherwise None.
    """
    for index, row in breakpoints.iterrows():
        if row['min'] <= value < row['max']:
            return index
    return None  # Return None if no group matches
def categorical_cuts(df, column_name):
    df['X_decile'] = df[column_name].astype(str)

    # Get unique categories of X_decile
    unique_deciles = df['X_decile'].unique()

    # Create an empty DataFrame with the specified columns and unique categories of X_decile
    columns = ['X_min', 'X_max', 'X_median', 'X_25%', 'X_75%']
    decile_stats = pd.DataFrame(index=unique_deciles, columns=columns)

    return decile_stats


def categorize_into_deciles_with_stats(df, column_name, Y, n_deciles=10, f_decile_tree =False):
    """
    Categorizes a specified column in a DataFrame into deciles (or specified quantiles) for numerical data,
    or uses existing categories for categorical data, and calculates various statistics for each group.

    Parameters:
    - df: Pandas DataFrame containing the data.
    - column_name: string, the name of the column to categorize.
    - Y: string, the name of another column to compute statistics for each group.
    - n_deciles: integer, default 10. Specifies the number of groups for numerical data.

    Returns:
    - DataFrame: Returns a DataFrame with group statistics.
    """
    # Check if the column is numeric or categorical
    if pd.api.types.is_numeric_dtype(df[column_name]):
        n_cuts = len(df[column_name].unique())
        print(n_cuts)
        
        # if too little numerical values, then it is treated as categorical
        if n_deciles>= n_cuts or n_cuts<=20:
            decile_stats = categorical_cuts(df, column_name)
        else:
            if f_decile_tree:
                df = numeric_treecuts(df, column_name, Y) 
            else:
                try: 
                    df = numeric_decilecuts(df, column_name, n_deciles=n_deciles-1)
                except:
                    df[column_name] =df[column_name].astype(str)
                    decile_stats = categorical_cuts(df, column_name)
            decile_stats = df.groupby('X_decile', observed=False)[column_name].agg([
            'min', 
            'max', 
            'median', 
            lambda x: x.quantile(0.25), 
            lambda x: x.quantile(0.75)
        ])
        # Rename the columns
        decile_stats.columns = ['X_min', 'X_max', 'X_median', 'X_25%', 'X_75%']
    
    else:
        
        decile_stats = categorical_cuts(df, column_name)
    decile_stats.reset_index()    

    # Calculate statistics for Y within each group
    y_stats = df.groupby('X_decile',dropna = False, observed=False)[Y].agg(['mean', 'std', 'median',
                                                            lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), 'count'])
    y_stats.columns = [f'{Y}_mean', f'{Y}_std', f'{Y}_median', f'{Y}_25%', f'{Y}_75%', 'n']
    total_sum = y_stats['n'].sum()
    y_stats['n_percentage'] = (y_stats['n'] / total_sum) * 100
    
    # Join decile_stats and y_stats
    combined_stats = decile_stats.join(y_stats)

    # Calculate overall mean for Y and discrepancy metrics
    overall_median_Y = df[Y].mean()
    combined_stats[f'gen_{Y}_mean'] = overall_median_Y
    combined_stats['discr'] = abs(combined_stats[f'{Y}_mean'] - overall_median_Y) / overall_median_Y
    combined_stats['max_discr'] = combined_stats['discr'].max()
    

    # Insert variable name at the start
    combined_stats.insert(0, 'varname', column_name)
    combined_stats.reset_index(inplace=True)
    combined_stats.columns = ['X_decile'] + combined_stats.columns[1:].tolist()
    combined_stats['X_min_str'] = combined_stats['X_min'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    combined_stats['X_max_str'] = combined_stats['X_max'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    # combined_stats.insert(0, 'X_decile', combined_stats.index)
    # Create x_string column based on the conditions
    combined_stats['x_string'] = np.where(
        combined_stats['X_min'].notna(),
        '[' + combined_stats['X_min_str'] + '-' + combined_stats['X_max_str'] + ']',
        combined_stats['X_decile']
    )
    combined_stats.drop(columns=['X_min_str',"X_max_str"], inplace=True)

    return combined_stats





# Updated plot_data_by_varname to work with subplots and add a table
def plot_data_by_varname(ax, df, var_name, Y):
    filtered_df = df[df['varname'] == var_name].copy()
    if filtered_df.empty:
        return

    filtered_df['x_string'] = filtered_df['x_string'].fillna('Missing')
    filtered_df['x_string'] = filtered_df['x_string'].astype(str)
    gen_y_mean = filtered_df[f'gen_{Y}_mean'].iloc[0]

    sns.lineplot(
        data=filtered_df,
        x='x_string',
        y=f'{Y}_mean',
        marker='o',
        linewidth=2.5,
        ax=ax,
        color='blue'
    )
    ax.axhline(y=gen_y_mean, color='red', linestyle='--', linewidth=1.5, label=f'Gen {Y} Mean')
    ax.set_title(f'Median {Y}: {var_name}', fontsize=12, fontweight='bold')
    ax.set_xlabel(var_name, fontsize=10)
    ax.set_ylabel(f'Median {Y}', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    sns.despine(ax=ax, offset=10, trim=True)

    # Add table below the plot
    table_data = filtered_df[['x_string', f'{Y}_mean', f'{Y}_std', f'{Y}_median', f'{Y}_25%', f'{Y}_75%', 'n']].values
    col_labels = ['x_string', f'{Y}_mean', f'{Y}_std', f'{Y}_median', f'{Y}_25%', f'{Y}_75%', 'n']
    table = Table(ax, bbox=[0, -0.3, 1, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    # Add column headers
    for i, label in enumerate(col_labels):
        table.add_cell(0, i, width=1.0/len(col_labels), height=0.1, text=label, loc='center', facecolor='lightgrey')

    # Add data rows
    for row_idx, row in enumerate(table_data):
        for col_idx, cell_value in enumerate(row):
            table.add_cell(row_idx+1, col_idx, width=1.0/len(col_labels), height=0.1, text=cell_value, loc='center')

    ax.add_table(table)
    ax.set_position([0.1, 0.4, 0.8, 0.5])  # Adjust plot position to make space for the table

