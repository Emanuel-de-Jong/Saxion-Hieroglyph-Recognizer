
# Introduction to machine learning - Saxon runes

Emanuel de Jong (495804) - Erik Markvoort (519894)

In this document will be found how the assignment progressed through the various versions as well as which important decisions were made during development.

## Versions

### 1: Feature extraction

1. Initial version
2. Removal of empty images
3. Adding HU moments, height to width ratio
4. Remove fully black images, documentation added

### 2: preprocessing

1. Initial version, compatible with 1.0
2. Saving scalers for web version, compatible with 1.1 and 1.2
3. 2.3: scaling features divided between minmax and z-score, documentation, compatible with 1.3

### 3: feature analysis

1. Initial version, compatible with 2.0
2. Changes in removing outliers, changes removal high correlation features, compatible with 2.1
3. Changes removal high correlation features, compatible with 2.2
4. Distribution analysis, compatible with 2.3

### 4: training and testing

1. Initial version, compatible with 3.0 and 3.1
2. Addition of kfold, compatible with 3.2
3. Fixes to kfold
4. Addition of gridsearchCV
5. Usage of pipeline

## Decision making

Custom features: the custom features that are made for our dataset are top to bottom ratio aswell as height to width ratio. Top to bottom ratio describes the open space at the top of the rune divided by the open space at the bottom of the rune. Height to width ratio describes the height divided by the width of the rune. Both of these features are telling of the shape of the drawn rune, which may be useful in the classification process.

GridsearchCV over ParameterGrid: both of these libraries are used for the same purpose; to find the ideal parameters for an algorithm to make the most accurately trained model. We learned that the use of GridsearchCV would also test the model with the KFold method, which is not only a requirement for the assignment but also a good method to prevent overfitting. Therefore GridsearchCV is used to find the ideal hyperparamters for each used algorithm. In notebook 4.3, Kfold is implemented manually which can be seen as an understanding of the method, where in 4.4 it is included in the use of GridsearchCV which makes the manual use redundant. if GridsearchCV was not used, a manual use of Kfold would be neccesary in order to prevent overfitting.

Usage of linear regression and extra trees as additional algorithms was decided based on the found accuracy of these models. Other algorithms that were considered included bagging and boosting, but did not have the desired results to justify using them. Extra trees specifically is an algorithm that builds onto the random forest algorithm which was found to already be quite effective.

Usage of Poetry: for the webserver, the same normalizers are needed as the ones fitted in the notebooks. So their objects are saved in the notebook and loaded in the webserver with pickle. This however means that both environments need to have the exact same python and package versions so the pickle can be read correctly. As the webserver uses Poetry, we decided to use the exact same Poetry project files for the notebooks too. You can set it up like this:
1. (If you don't yet have Poetry) `pip install poetry`
2. `poetry install`
3. Select the venv made by Poetry in the notebook. It will be something like `student-template-abcd1234`.

## Conclusion

Between CSV, KNN, decision tree, naïve bayes, random forest, logistic regression and extra trees, the parameters found with gridsearchCV are used to make a list of models and the most accurate one is automatically selected to be saved. The algorithm found in this manner is an SVM model with the parameters: (break_ties=True, coef0=1.0, degree=5, kernel='poly', probability=True) which results in a total average of around 84%, which is a satisfactory result.
