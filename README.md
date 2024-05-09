The Jupyter Notebook demonstrates three experiments:

Creation of multivariate Gaussian distributions for two classes: "Resting" and "Stressed".
Resting class: Mean ([60,10]), Standard Deviation ([[20,100],[100,20]])
Stressed class: Mean ([100,80]), Standard Deviation ([[50,20],[20,50]])

1) Development of a K Means Classifier for unsupervised classification of the normally distributed data called from classifiers.py.

2) Design of a Principal Component Analysis method to reduce feature size on a plane for the iris dataset called from feature_reduction.py.

3) Performance of Data Wrangling with the Pokemon Dataset and prediction of the legendary nature of Pokemons based on their numerical and categorical features.
- Data cleaning included the utilization of frequency encoding to categorize missing categorical columns.
- Post cleaning, a Random Forest Classifier was employed to predict missing values of the categorical classes, providing a more non-linear approach to filling the values.
- Additionally, a Support Vector Classifier (SVC) was implemented to predict the binary nature of the legendary target variable.
- 'Linear' and 'rbf' kernel hyperparameter tuning was applied to the trained model using GridSearchCV.
