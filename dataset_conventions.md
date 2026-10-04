As we are in the exploratory phase in our project, i think it would make sense to have a general set of rules for our different types of datasets.

1. Datasets should be cleaned for missing values before they are fed into the program.
2. Datasets do not need to be standardized or normalized. Scaling happens in utils/dataset.build_datasets.
3. Datasets should contain a "date" column in the format (YYYY-MM-DD). This might be generalized later for order book data.
4. Features at row t should onlt contain information available at the end of day t.
5. Datasets should specifically contain a day t's closing price at row t, as we are going to be computing our label from it.
6. The label 'target' at day t is the return from day t to day t+1. Calculated with: y_t = (close_t+1 / close_t) - 1. 
---
