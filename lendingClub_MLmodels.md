
# MULTIPLE LINEAR REGRESSION - PREDICTING INTEREST RATE

##  1. BUSINESS UNDERSTANDING

#### Our main motto is to build an online tool to offer potential customer an approximate interest rate that they would get based on various variables such as Purpose, Annual income, repayment term and employment length etc. 

#### Building a machine learning model from the historical data to predict the potential interest rate.

#### This could be a strategy to increase the existing customer base by providing this online checking tool.

## Import the dataset


```python
data = spark.read.format("com.databricks.spark.csv")\
    .option("header","true")\
    .option("inferSchema","true")\
    .load("hdfs://localhost:8020/user/master/Loan_reduced.csv")
```


```python
print type(data)
```

## 2. DATA UNDERSTANDING


```python
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.rcParams['figure.figsize'] = (12,8)
import pandas as pd
data.toPandas().head(5)
```


```python
data.count()
```


```python
data.printSchema()
```


```python
data.createOrReplaceTempView("data")
```


```python
data.toPandas().describe()
```

#### Since int_rate is string, we cannot check the descriptive statistics for the variable. We have to convert int_rate into double in order to view the basic statistics.


```python
from pyspark.sql.functions import *
data = data.withColumn('int_rate', regexp_replace('int_rate', '%',''))
# We remove the % symbol from the data as it considers % as string value.
```


```python
from pyspark.sql.types import DoubleType
data = data.withColumn("int_rate", data["int_rate"].cast(DoubleType()))
```


```python
data.createOrReplaceTempView("data")
```


```python
data.printSchema()
```


```python
data.toPandas().describe()
```

#### Now the data type of int_rate is changed to double and we could see the descriptive statistics.


```python
## Set target variable's name to label
data_redu = spark.sql("SELECT int_rate as label, loan_amnt, term, emp_length, home_ownership,\
                    annual_inc, verification_status, purpose, open_acc FROM data")
## Register as Spark SQL
data_redu.createOrReplaceTempView("data_redu")
```


```python
spark.sql("SELECT * FROM data_redu")
```


```python
data_redu.toPandas().head(5)
```

## Checking descriptive statistics and frequencies

#### Descriptive statistics for continuous variables 
#### Frequency tables for the categorical variables


```python
data_redu.toPandas().describe()
```


```python
spark.sql("SELECT purpose, COUNT(*) AS cnt FROM data_redu GROUP BY purpose ORDER BY cnt DESC").toPandas()
```


```python
spark.sql("SELECT verification_status, COUNT(*) AS cnt FROM data_redu GROUP BY verification_status ORDER BY cnt DESC").toPandas()
```


```python
spark.sql("SELECT home_ownership, COUNT(*) AS cnt FROM data_redu GROUP BY home_ownership ORDER BY cnt DESC").toPandas()
```


```python
spark.sql("SELECT term, COUNT(*) AS cnt FROM data_redu GROUP BY term ORDER BY cnt DESC").toPandas()
```


```python
spark.sql("SELECT emp_length, COUNT(*) AS cnt FROM data_redu GROUP BY emp_length ORDER BY cnt DESC").toPandas()
```

## Data Visualization


```python
from ggplot import *
```


```python
data_p = data_redu.toPandas()
```


```python
p = ggplot(data_p, aes('label')) + geom_bar()
display(p)
```


```python
p = ggplot(data_p, aes('annual_inc', 'label')) + geom_point()
display(p)
```


```python
p = ggplot(data_p, aes('loan_amnt', 'label')) + geom_point()
display(p)
```


```python
p = ggplot(data_p, aes('open_acc', 'label')) + geom_point()
display(p)
```

## 3. DATA PREPARATION

## StringIndexer & OneHotEncoder

In algorithms like Linear Regression Analysis or Logistic Regression, categorical variables have to be transformed into numerical variables.

In general, there are two ways to do this:
1. Category Indexing using `StringIndexer`: This is basically assigning a numeric value to each category from {0, 1, 2, ... numCategories-1}. This produces an implicit ordering among the categories, and is more suitable for ordinal variables (e.g., Not Satisfied: 0, Neutral: 1, Satisfied: 2)
2. One-Hot Encoding using `OneHotEncoder`: This converts categories into binary vectors with at most one nonzero value (eg: (Blue: [1,0]), (Green:[0,1]), (Red:[0,0])

#### Please note: If you want One-Hot Encoding, first use Category Indexing and then apply One-Hot Encoding


```python
# Import the functions OneHotEncoder and StringIndexer
from pyspark.ml.feature import OneHotEncoder, StringIndexer
```


```python
stringIndexer = StringIndexer(inputCol="term", outputCol = "termIndex")
model = stringIndexer.fit(data_redu)
data_redu = model.transform(data_redu)
```


```python
encoder = OneHotEncoder (inputCol="termIndex", outputCol="termVec")
data_redu = encoder.transform(data_redu)
```


```python
stringIndexer = StringIndexer(inputCol="emp_length", outputCol = "emp_lengthIndex")
model = stringIndexer.fit(data_redu)
data_redu = model.transform(data_redu)
```


```python
encoder = OneHotEncoder (inputCol="emp_lengthIndex", outputCol="emp_lengthVec")
data_redu = encoder.transform(data_redu)
```


```python
stringIndexer = StringIndexer(inputCol="home_ownership", outputCol = "home_ownershipIndex")
model = stringIndexer.fit(data_redu)
data_redu = model.transform(data_redu)
```


```python
encoder = OneHotEncoder (inputCol="home_ownershipIndex", outputCol="home_ownershipVec")
data_redu = encoder.transform(data_redu)
```


```python
stringIndexer = StringIndexer(inputCol="verification_status", outputCol = "verification_statusIndex")
model = stringIndexer.fit(data_redu)
data_redu = model.transform(data_redu)
```


```python
encoder = OneHotEncoder (inputCol="verification_statusIndex", outputCol="verification_statusVec")
data_redu = encoder.transform(data_redu)
```


```python
stringIndexer = StringIndexer(inputCol="purpose", outputCol = "purposeIndex")
model = stringIndexer.fit(data_redu)
data_redu = model.transform(data_redu)
```


```python
encoder = OneHotEncoder (inputCol="purposeIndex", outputCol="purposeVec")
data_redu = encoder.transform(data_redu)
```

## VectorAssembler

`VectorAssembler` is a transformer that combines a given list of columns into a single vector column.

It is useful for combining raw features and features generated by different feature transformers into a single feature vector, in order to train ML models like Logistic Regression, Linear Regression or Decision Trees. 


```python
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
```


```python
assembler = VectorAssembler(
    inputCols = ["loan_amnt","annual_inc", "open_acc", "termVec", "emp_lengthVec", "home_ownershipVec","verification_statusVec", "purposeVec"],
    outputCol = "features")
```


```python
output = assembler.transform(data_redu)
```


```python
print type(output)
```


```python
output.toPandas().head(4)
```


```python
output.select("features","label").show(5, truncate = False)
```

## Feature Scaling

Feature scaling is a method used to standardize the range of independent variables or features of data. 


```python
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StandardScaler
from pyspark.sql.functions import *
```


```python
scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
```


```python
scalerModel = scaler.fit(output)
```


```python
scaledData = scalerModel.transform(output)
```


```python
print("Features scaled to range: [%f, %f]" % (scaler.getMin(), scaler.getMax()))
scaledData.select("features", "scaledFeatures").show()
```


```python
scaledData = scaledData.drop('features')
```


```python
scaledData = scaledData.select(col("scaledFeatures").alias("features"), col("label").alias("label"))
```


```python
scaledData.show(5)
```

### Split the dataset in training and test-dataset!


```python
# Randomly split data into training and test sets. 
# Set seed for reproducibility
(trainingData, testData) = scaledData.randomSplit([0.7,0.3], seed = 111)
```


```python
print trainingData.count()
print testData.count()
```

# 4. Modelling

## Model 1 - Multivariate Regression model


```python
# Import LinearRegression class
from pyspark.ml.regression import LinearRegression
```


```python
# Define LinearRegression algorithm
mlr = LinearRegression()
```


```python
# Fit base model
modelA = mlr.fit(trainingData, {mlr.regParam:50.0})
```


```python
print(modelA.intercept)
print(modelA.coefficients)
```


```python
predictionsA = modelA.transform(testData)
predictionsA.show(5)
```

### Evaluation of Linear Model 1

#### Root Mean Squared Error


```python
from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(metricName="rmse")

RMSE = evaluator.evaluate(predictionsA)
print("ModelA: Root Mean Squared Error = " + str(RMSE))
```

#### Coefficient of Determination - R2


```python
modelA.summary.r2
```

## Model 2- Polynomial Regression model


```python
from pyspark.ml.feature import PolynomialExpansion
```


```python
polyExpansion = PolynomialExpansion(degree=2, inputCol="features", outputCol="features_poly")
```


```python
polyDF = polyExpansion.transform(scaledData)
```


```python
(trainingData1, testData1) = polyDF.randomSplit([0.7,0.3], seed = 111)
```


```python
print trainingData1.count()
print testData1.count()
```


```python
mlr = LinearRegression()
```


```python
# Fit a model
model_poly = mlr.fit(trainingData1)
```


```python
print(model_poly.intercept)
print(model_poly.coefficients)
```


```python
# Make predictions
predictions_poly_A = model_poly.transform(testData1)
predictions_poly_A.show(5)
```

### Evaluation of Polynomial Model 2


```python
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(metricName="rmse")

RMSE = evaluator.evaluate(predictions_poly_A)
print("Model_poly: Root Mean Squared Error = " + str(RMSE))
```


```python
model_poly.summary.r2
```

## Model 3 - Tuning Model 2 


```python
##=====build cross valiation model======

# estimator
lr = LinearRegression(featuresCol = 'features', labelCol = 'label')

# parameter grid
from pyspark.ml.tuning import ParamGridBuilder
param_grid = ParamGridBuilder().\
    addGrid(lr.regParam, [0, 0.5, 1]).\
    addGrid(lr.elasticNetParam, [0, 0.5, 1]).\
    build()
    
# evaluator
evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='label', metricName='r2')

# cross-validation model
from pyspark.ml.tuning import CrossValidator
cv = CrossValidator(estimator=lr, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=4)
```


```python
cv_model = cv.fit(trainingData1)
```


```python
pred_training_cv = cv_model.transform(trainingData1)
pred_test_cv = cv_model.transform(testData1)
```


```python
# performance on training data
evaluator.evaluate(pred_training_cv)

# performance on test data
evaluator.evaluate(pred_test_cv)
```


```python
print('best regParam: ' + str(cv_model.bestModel._java_obj.getRegParam()) + "\n" +
     'best ElasticNetParam:' + str(cv_model.bestModel._java_obj.getElasticNetParam()))
```

#### Based on the above parameter, tune the polynomial model to obtain optimum results for the model


```python
# Fit a model
model_final = mlr.fit(trainingData1, {mlr.regParam:0.0})
```


```python
print(model_final.intercept)
print(model_final.coefficients)
```


```python
# Make predictions
predictions_final = model_poly.transform(testData1)
predictions_final.show(5)
```

#### Evaluation of the tuned Polynomial Model 3


```python
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(metricName="rmse")

RMSE = evaluator.evaluate(predictions_poly_A)
print("ModelA: Root Mean Squared Error = " + str(RMSE))
```


```python
model_final.summary.r2
```

# 4. Evaluation

Upon performing modelling on different training set, we have the RMSE and R2 values for all our models.
Lower the RMSE value, better the model.
Higher the value of R2, better the model.

Looking at our models, Polynomial Regression model and Tuned Polinomial Regression model give us the same RMSE and R2 values.

For the simplicity of the model, we use Polynomial model as our final model.

### Compare training- and test-fit for the best model (model 2)

Make predictions based on the training dataset and testing set


```python
# Model 
predictions_trainpoly_A = modelA.transform(trainingData1)
predictions_testpoly_A = modelA.transform(testData1)
```

Calculate and Compare R2 of both models


```python
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(metricName="r2")
```


```python
# Model_poly
r2_trainingp_A = evaluator.evaluate(predictions_trainpoly_A)
r2_testp_A = evaluator.evaluate(predictions_testpoly_A)
```


```python
print r2_trainingp_A *100
print r2_testp_A *100
print (r2_trainingp_A - r2_testp_A) * 100
```

## Inspect the main assumption of regression models - Normal Distribution of Error


```python
from ggplot import *
from seaborn import *
from pyspark.sql.functions import *
```


```python
## Calculate the residuals
# In dataset: labels (observed values) and predicted values (based on model) --> difference of both: residual
#predictionsA.show(5)
#residuals_A = predictionsA.select(col('label') - col('prediction') )
#residuals_A.show(5)
predictions_poly_A = predictions_poly_A.withColumn("residuals", col('label') - col('prediction'))
predictions_poly_A.show(5)
```


```python
print type(predictions_poly_A)
data.createOrReplaceTempView("predictionsA")
```


```python
## Convert Spark DataFrame into pandas
predictions_poly_A_pandas = predictions_poly_A.toPandas()
print type(predictions_poly_A_pandas)
```


```python
from seaborn import *
```


```python
distplot(predictions_poly_A_pandas['residuals'], rug = True)
```

# 5. Deployment 

Deployment is where machine learning pays off.

In this final phase of the Cross-Industry Standard Process for Data Mining (CRISP-DM) process, it doesn't matter how brilliant the discoveries may be, or how perfectly your models fit the data, if we don't actually use those things to improve the way that you do business.

#### So final deployment of the model has to be carried out by implementing the tool online in the Lending club website.
