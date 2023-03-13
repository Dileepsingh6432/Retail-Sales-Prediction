
[Dataset](https://www.kaggle.com/competitions/rossmann-store-sales/data)

### Retail-Sales-Prediction
The objective of this project is to forecast or predict the sales. This is based on a Rossman Store's Data, and we use different regression algorithms to predict the sales.
# Introduction
Rossman operates over 3,000 drug stores in 7 European countries. Currently, Rossman store managers are tasked with predicting their daily sales for up to six weeks in advance. Store sales are influenced by many factors, including promotions, competition, school and state holidays, seasonality, and locality. With thousands of individual managers predicting sales based on their unique circumstances, the accuracy of results can be quite varied.

# Problem Statement

Rossman store managers are tasked with predicting their daily sales for up to six weeks in advance. Store sales are influenced by many factors, including promotions, competition, school and state holidays, seasonality, and locality. With thousands of individual managers predicting sales based on their unique circumstances, the accuracy of results can be quite varied.
You are provided with historical sales data for 1,115 Rossman stores. The task is to forecast the "Sales" column for the test set.

# Data Summary
The given dataset is a dataset from Rossman industry who operates over 3,000 drug stores in 7 European countries, and we have to analyze the sales of the stores and the factors affecting the sales. In the given dataset, it has 1017209 rows and 18 column.  And there are some missing values, and there is no duplicate values in the dataset.
There are two datasets we are dealing in this project:
**sales_data:** It contains all the information regarding store sales, Day of Week, Date of sales, number of customers, stores is open or closed, stores are running promo or not, school holiday or not , and state holiday or not.
**store_data:** It contains all the information regarding store type, Assortment, Competition distance, Competition open since month, Promo 2 Since Week, Competition Open Since Year, Promo 2, Promo 2 Since Year, Promo Interval.

# Variables Description
**Id** - an Id that represents a (Store, Date) duple within the test set
**Store** - a unique Id for each store
**Sales** - the turnover for any given day (this is what you are predicting)
**Customers** - the number of customers on a given day
**Open** - an indicator for whether the store was open: 0 = closed, 1 = open
**StateHoliday** - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None
**SchoolHoliday** - indicates if the (Store, Date) was affected by the closure of public schools
**StoreType** - differentiates between 4 different store models: a, b, c, d
**Assortment** - describes an assortment level: a = basic, b = extra, c = extended
**CompetitionDistance** - distance in meters to the nearest competitor store
**CompetitionOpenSince[Month/Year]** - gives the approximate year and month of the time the nearest competitor was opened
**Promo** - indicates whether a store is running a promo on that day
**Promo2** - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
**Promo2Since[Year/Week]** - describes the year and calendar week when the store started participating in Promo2
**PromoInterval** - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store



# Data Wrangling
The data wrangling is performed using the numpy, pandas, etc. to make the data ready for analysis.
# Data visualization 
The following analysis is done using the different charts considering the univariate, bivariate, and multivariate analysis.
* Analysis  based on Store type
* Analysis based on Assortment levels
* Analysis based on store open or closed on State Holiday
* Bar Plot of various categorical variable the basis of Sales (Bivariate)
* Assortment on the basis of Sales (Bivariate)
* Scatter plot b/w Sales and competition distance considering the Store Type (Multivariate)
* Plot of Sales corresponding to Date (Bivariate)
* Stacked Bar Plot of store type and promo data on the basis of sales (Multivariate)
* Average sales of each store type with respect to each Assortment strategies.
* Multiple Line plot indicating the sales throughout the year (Multivariate)
* Relation between Store Type and state Holiday on the basis of Sales (Multivariate)
* Relation between Customers and Store Type on the basis of Sales (Multivariate)
* Stacked bar plot showing total sales of each month of each year.
* Correlation Heat map
* Pair Plot
# Feature Engineering and Data Preprocessing

**Handling Missing Values**
The missing values in the dataset is handled in appropriate ways considering the feature (categorical or numerical), as follows.
CompetitionDistance' shows an positive or right skewed data so, we replace the null values with the median because the median is not affected by the outliers.

'CompetitionOpenSinceMonth' and 'CompetitionOpenSinceYear' both are categorical features so, we replaced the null values with the mode.

In 'Promo2SinceWeek', 'Promo2SinceYear', and 'PromoInterval' missing values represent that there are no promo2 (promo in consecutive month) in the particular date. So, we replace the null value with zero.

**Handling Outliers**

here are only 0.95% (approx) outliers present in the data.

Here, the number of sales outliers is 9728, in which 6995 observations is present in the dataframe when the store is running a promo. So, it will not be a good idea to treat outliers.

So, we will not drop any observation for further pre-processing.

**Categorical Encoding**

In 'StateHoliday' Column, there are different types of hildays so, we replaced all holidays with 1.

There are three categorical column present in the dataframe (i.e 'StoreType' , 'Assortment', and 'PromoInterval'). Now, we created a column called 'promo2running' which is 1 if sale month is present in the promointerval month else 0.

We changed two categorical columns (i.e 'StoreType' and 'Assortment') into numerical ones by creating the dummy variables.

**Feature Manipulation & Selection**

Here, we converted two columns into a single one because 'Promo2SinceYear' and 'Promo2SinceWeek' represents that since when the promo2 is running. We created a column 'Promo2Open' which defines the number of month since the promo2 is running.

We dropped 'Promo2SinceYear' and 'Promo2SinceWeek' columns from the dataset becaused we created 'Promo2Open' by using them.

Here, we are combining 'CompetitionOpenSinceYear' and 'CompetitionOpenSinceMonth' into 'CompetitionOpen', and 'CompetitionOpen' depicts since how many months the competition store is running. We did the same thing as we did above. After combining both columns 'CompetitionOpenSinceYear' and 'CompetitionOpenSinceMonth', we dropped these columns from dataset.

Here, we removed four columns (i.e ['Store', 'year', 'WeekOfYear', 'DayOfYear', 'Date']) from our dataset which is irrelavant for our models. Because we will not give these input variables to ML model for sales prediction.

We converted two columns into a single one because 'Promo2SinceYear' and 'Promo2SinceWeek' represents that since when the promo2 is running. We created a column 'Promo2Open' which defines the number of month since the promo2 is running.

We dropped 'Promo2SinceYear' and 'Promo2SinceWeek' columns from the dataset becaused we created 'Promo2Open' by using them.

We are combined 'CompetitionOpenSinceYear' and 'CompetitionOpenSinceMonth' into 'CompetitionOpen', and 'CompetitionOpen' depicts since how many months the competition store is running. We did the same thing as we did above. After combining both columns 'CompetitionOpenSinceYear' and 'CompetitionOpenSinceMonth', we dropped these columns from dataset.

we removed four columns (i.e ['Store', 'year', 'WeekOfYear', 'DayOfYear', 'Date']) from our dataset which is irrelavant for our models. Because we will not give these input variables to ML model for sales prediction.

Sales is our target variable and rest are the independent variables or we can say that these are the features which influencing the target variables.
**Data Transformation**
We have used MinMaxScaler to scale my data. Min-max scaler is a method for feature scaling, which is a technique used to normalize the range of the independent variables (features) of a dataset. The goal of min-max scaling is to transform the features such that they are in the range of [0, 1].

X_scaled = (X - Xmin) / (Xmax - Xmin)

Where X is the original feature value, Xmin is the minimum value of the feature and Xmax is the maximum value of the feature. This method is particularly useful for algorithms that are sensitive to the scale of the input features, such as k-nearest neighbors and artificial neural networks.

We used minmax scaler to scale our data because all values will fall between 0 and 1. while the range is increased while using the standard scaler because it falls between the min and max values. Our data does not follow normal distribution so, we used minmax scaler instead of standard scaler.

We have used this method after spliting our data.

**Dimesionality Reduction**
Dimensionality reduction is not always necessary in regression, as the model can handle high dimensional input data. However, it may be useful in certain situations such as reducing the computational cost of training and testing the model, removing noise or irrelevant features from the input data, or improving the interpretability of the model by identifying the most important features. It can also be useful in preventing overfitting, which occurs when a model is too complex for the amount of data it is being trained on.

**Data Splitting**
We take last six months of data for testing, and the remaining for training the model.

**Handling Imbalanced Dataset**


No need to think the dataset is imbalanced.

Handling imbalanced datasets refers to the process of addressing the issue of unequal distribution of classes in a dataset. Imbalanced datasets occur when one class has significantly more samples than other classes, which can result in poor performance of machine learning models. There are several techniques that can be used to handle imbalanced datasets.

# Model Implementation
**Linear Regression**

It gives 79% accuracy on the test data (i.e r2_score is 0.790007 for test data). And we get 81% accuracy (r2_score = 0.8155) after using the cross-validation.

**Decision Tree**
It gives 87% accuracy on the test data (i.e r2_score is 0.8720 for test data). And 88% (i.e. r2_score = 0.8843) after tunning hyperparameter.

**Random Forest Regressor**

The random forest regressiore provides 93.77% accuracy because it is gives 0.9377 r2_score.

**L1 and L2 Regularization**
Lasso gives 14% accuracy while ridge provides approximately 79% accuracy. And elastic net is worst model for this data because it gives 9% accuracy.

After hyperparameter tunning, Lasso gives 79% (approximately) accuracy.

* Among the all regression models, it is clear that Random Forest Regressor is giving the best result with the accuracy of 93.6% followed by Decison Tree Regressor with accuracy of 87.2%. So, we will use the random forest regressor to predict the sales.
