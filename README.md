# Bivariate Analysis Library

## Description
The **Bivariate Analysis Library** is a Python package designed for analyzing tabular datasets. It enables automatic binning of data for either binary classification or regression tasks, providing insights into the most discriminative information for each feature. The library generates simple, interpretable tables and visualizations for better understanding of your dataset.

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Documentation](#documentation)
5. [Technologies](#technologies)
6. [Testing](#testing)
7. [Deployment](#deployment)
8. [Credits](#credits)
9. [License](#license)
10. [Contact](#contact)

## Features
- Automatic binning for binary classification and regression tasks.
- Binning methods: simple decision trees (slower but better) or manual binning.
- Generates a comprehensive table summarizing the most discriminative bin for each feature.
- Visualization tools for exploring feature statistics and distributions.

## Installation
To install the library, clone the repository and set up the environment:

```bash
# Clone the repository
git clone https://github.com/jackmat/bivariate_analysis.git

# Navigate to the directory
cd bivariate_analysis

# Install required dependencies
pip install -r requirements.txt
```

## Usage
Below is an example of how to use the library for analyzing a dataset:

```python
import pandas as pd
from sklearn.datasets import load_iris
from src.bivariate_functions import categorize_into_deciles_with_stats, plot_data_by_varname

def main(df, target_column, drop_cols, n_deciles=10, f_decile_tree=False):
    # Validate input columns
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is not in the DataFrame.")
    for col in drop_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' in drop_cols is not in the DataFrame.")

    bivariate_statistics = pd.DataFrame()
    visualizations = {}
    cols_to_evaluate = list(set(df.columns) - set([target_column]) - set(drop_cols))

    for column in cols_to_evaluate:
        result = categorize_into_deciles_with_stats(
            df, column, target_column, n_deciles=n_deciles, f_decile_tree=f_decile_tree
        )
        bivariate_statistics = pd.concat([bivariate_statistics, result], ignore_index=True)

    for var in cols_to_evaluate:
        visualizations[var] = plot_data_by_varname(bivariate_statistics, var, target_column)

    return bivariate_statistics, visualizations

if __name__ == "__main__":
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    target_column = "sepal length (cm)"
    drop_cols = []

    results, visualizations = main(df, target_column, drop_cols)
    print("Bivariate analysis completed.")
    print(results)
```

## Documentation
For detailed documentation, visit the [GitHub Repository](https://github.com/jackmat/bivariate_analysis).

### Key Functions
1. `categorize_into_deciles_with_stats`:
   - Categorizes data into bins (e.g., deciles) and computes statistical information.
2. `plot_data_by_varname`:
   - Visualizes the distribution and discriminative power of features.

## Technologies
- **Python**: Core programming language.
- **Pandas**: For data manipulation.
- **Matplotlib**: For data visualization.
- **Scikit-learn**: For decision tree-based binning.

## Testing
Run unit tests to validate the library (No tests have been created):

```bash
# Run tests using pytest
pytest tests/
```

## Deployment
This library can be deployed locally or integrated into larger data analysis pipelines. Ensure all dependencies are installed, and use the provided API for seamless integration.

## Credits
- **Author**: [Jackmat](https://github.com/jackmat)
- Special thanks to contributors and the open-source community.

## License
This project is licensed open sourced.


## Contact
For any questions or support, reach out via:
- **GitHub Issues**: [Submit an issue](https://github.com/jackmat/bivariate_analysis/issues)

---

Happy analyzing!

