The jupyter notebook showcases three experiments.

Creating multivariate Gaussian distributions of two classes - resting and stressed.
Resting class - Mean ([60,10]), Standard Deviation ([[20,100],[100,20]])
Stressed Class = Mean (100,80]), Standard Deviation ([50,20],[20,50]])

1) Developing a K Means Classifier for unsupervised classification of the normally distributed data

2) Designing a Principal Component Analysis method to reduce feature size on plane for the iris dataset

3) Performing Data Wrangling with the Pokemon Dataset and predicting the legendary nature of Pokemons based on their numerical and categorical features.

The data cleaning included usig a distribution called frequency encoding to categorize the missing categorical columns. Post cleaning, a Random Forest Classifier was used to predict the missing values of the categorical classes, thereby providing a more non-linear approach to filling the values.

Furthermore, implemented a Support Vector Classifier (SVC) to predict the binary nature of the legendary target variable, whilst applying 'linear' and 'rbf' kernel hyperparameter tuning using GridSearchCV to the trained model.
