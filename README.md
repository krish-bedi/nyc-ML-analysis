
# NYC Leading Causes of Death Analysis

## Project Overview
This project conducts a comprehensive analysis of the leading causes of death in New York City. Utilizing a dataset that covers various demographics and causes, the analysis employs advanced statistical and machine learning techniques to uncover patterns and insights related to mortality rates. The project is split into two main components: a machine learning model to predict death counts (`nyc-test.py`) and a series of data visualizations to interpret the data (`nyc-visual.py`).

## Technical Stack
- **Programming Language**: Python
- **Libraries Used**:
  - `pandas` for data manipulation
  - `scikit-learn` for machine learning
  - `matplotlib` and `seaborn` for data visualization

## Installation and Execution
1. Clone the repository to your local machine.
2. Ensure Python is installed along with the necessary libraries: pandas, scikit-learn, matplotlib, and seaborn.
3. Run `nyc-test.py` to execute the machine learning model and view the predictive performance.
4. Run `nyc-visual.py` to generate and view the data visualizations.

## Features

### Predictive Modeling (`nyc-test.py`)
- Utilizes a RandomForestRegressor to model the relationship between demographic factors and death counts.
- Employs GridSearchCV for hyperparameter tuning to optimize model performance.
- Evaluates model accuracy with the R-squared metric, achieving an impressive score indicating a high level of predictive reliability.
- Presents feature importance to highlight which factors most significantly impact death counts.

### Data Visualization (`nyc-visual.py`)
- Provides a histogram of the age-adjusted death rate to understand its distribution.
- Creates a bar chart to compare the count of deaths by different leading causes.
- Uses a boxplot to depict the age-adjusted death rate across different races and ethnicities.
- Generates a line chart to visualize the trend of deaths over time, offering insights into how mortality rates have evolved.

## Results
The analysis revealed a high R-squared score of 0.954, indicating a strong model fit. The model's best parameters were a max depth of 5 and 300 estimators, suggesting an optimal balance between bias and variance.

## Conclusion
This project not only provides a deep analytical view of the leading causes of death in New York City but also showcases the application of machine learning in public health. The insights gained can help in strategic planning and policy-making to address public health concerns.

## Author
- Krish Bedi

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
