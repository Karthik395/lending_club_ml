# lending_club_ml

The main aim of this excercise was to predict the probable interest rates by analyzing the figures of 2017 Q4 of lending club.

The main dataset was dataset was downloaded from Lending club website.

The dataset was reduced with only the importnat variabes that help of to predict the interest rates and prevent overfitting.

Finally the analyzed dataset had 105451 records and 12 variables

The accuracy of the model and other important details are provided in the presentation attached.

Step by step analysis :

Variables - loan_amnt, term, emp_length, home_ownership, annual_inc, verification_status, purpose, addr_state, open_acc

1. Data understanding - Visualize using pandas, descriptives and frequencies, different plots using seaborn and ggplot.

2. Data preparation - removing % symbol from the target variable, String indexer(giving a value to cat variables, OneHotEncoder(Dummies), Vector assembler, feature scaling, Split into training and test

3. Modelling - Multivariate regression model, Polynomial regression analysis, Tuning the model with hyper parameters

4. Evaluation - RMSE and R2 values, Normal distribution of error (bell shape curve should be evident)
