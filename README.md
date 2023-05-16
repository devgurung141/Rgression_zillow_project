# Home Value Prediction at Zilllow

# Project Description
The goal of the project is to analyze data, find key drivers of property value for single family properties, construct Machilne Learning Regressin Model to predict home value , and recommend ways to make a better model. 

# Project Goals
* Find key drivers of home value
* Use drivers to develop Learning Regressin Model to predict home value
* Offer recommendations to make a better model

# Initial Questions
* Does location have affect on home value?
* What is bathroom and bedroom role on home value?
* Does a home with one bathroom is more expesnive than a home with two bedrooms?
* What is relation of area to home value?

# The Plan

* Acquire data
    * Acquire data from zillow database from Codeup database using MySQL queryy using function from wrangle.py file

* Prepare data
    * Use functions from wrangle.py to clean data. 
      * Rename column names
      * Replaced fips with county names
      * Remove outliers
      * Encode attributes to fit in ML format.
    * split data into train, validate and test (approximatley 56/24/20)

* Explore Data
    * Use graph and hypothesis testing to find driving factors of home value, and answer the following initial questions
        * Does location have affect on home value?
        * What is bathroom and bedroom role on home value?
        * Does a home with one bathroom is more expesnive than a home with two bedrooms?
        * What is relation of area to home value?

* Develop Model
    * Use MinMaxScaler to scale data
    * Set up baseline RMSE
    * Evaluate models on train data and validate data
    * Select the best model based on the RMSE
    * Evaluate the best model on test data to make predictions

* Draw Conclusions

# Data Dictionary
| Feature | Definition |
|:--------|:-----------|
| bathroomcnt| Number of bathrooms in home including fractional bathrooms|
| bedroomcnt| NUmber of bedrooms in home|
| calcualtedfinishedsquarefee| Calculated total finished living area of of the home|
| taxvaluedollarcnt| The total tax assessed value of the parcel|
| fips| Federal Information Processing Standard code|
| parcelid| Unique identifier for parcels (lots)|


# Steps to Reproduce
1. Clone this repo 
2. To acquire data, need to have access to to MySQL database of codeup. 
3. Data can be also be acquired from [Kaggle](https://www.kaggle.com/competitions/zillow-prize-1/data), save a file as 'zillow_2017.csv', and put the file into the cloned repo 
5. Run notebook

# Takeaways and Conclusions
* bathroom, bedroom, sqft, location are driving attributes of home value.
* Higher number of bathroom and bedroom means higher home value.
* Increase in area leads in increase of home value.
* Los Angeles county has the home with less value that ventura and orange county
* bathroom and sqft have high corelation with our target.
* Polynomial Regression with degree of 3 resulted in an improved RMSE of 243241.06 on out of sample data, outperforming the baseline RMSE of 246241.41 on training data.
# Recommendations
* explore other attributes to see thier effects on target variables
* use differnt model to improve predictions..
