# 1. DATA COLLECTION
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('ml-titanic').getOrCreate()
df = spark.read.csv("/user/hadoop/train.csv", header = True, inferSchema = True)

# 2. DATA PROFILING
# See schema of the data
df.printSchema()  
df.show(5)

# Count of rows and columns
rows = df.count()
print('N rows = '+str(rows))
columns = len(df.columns)
print('N columns = '+str(columns))

# Balance of target variable
df.groupBy(df['Survived']).count().show()

# We don't analyze Name and PassengerID since they are identifiers, not features

# Numeric features 
numeric_features = ['Age', 'Fare']
# Summary statistics for numeric features
df.select(numeric_features).describe().show()

# Analize percentiles and outliers on Age and Fare. 

# 25th and 75th percentile of Age column
df.selectExpr('percentile(Age, 0.25)').show()
df.selectExpr('percentile(Age, 0.75)').show()
        
# Outliers in Age column 
df.filter((df['Age']<  -6.6875) | (df['Age']> 64.8125)).show()
df.filter((df['Age']<  -6.6875) | (df['Age']> 64.8125)).count()

# 25th and 75th percentile of Fare column
df.selectExpr('percentile(Fare, 0.25)').show()
df.selectExpr('percentile(Fare, 0.75)').show()

# Outliers in Fare column
df.filter((df['Fare']< -26.724) | (df['Fare']> 65.6344)).show()
df.filter((df['Fare']< -26.724) | (df['Fare']> 65.6344)).count()

# Categorical features
# Count distinct values
df.groupBy(df['Sex']).count().show()
df.groupBy(df['Sex']).count().count()
df.groupBy(df['Pclass']).count().show()
df.groupBy(df['Pclass']).count().count()
df.groupBy(df['Embarked']).count().show()
df.groupBy(df['Embarked']).count().count()
df.groupBy(df['Ticket']).count().show()
df.groupBy(df['Ticket']).count().count()
df.groupBy(df['Cabin']).count().show()
df.groupBy(df['Cabin']).count().count()
df.groupBy(df['SibSp']).count().show()
df.groupBy(df['SibSp']).count().count()
df.groupBy(df['Parch']).count().show()
df.groupBy(df['Parch']).count().count()

# Count of missing values for each column
from pyspark.sql.functions import isnan, when, count, col
df.select([count(when(col(c).isNull(),c)).alias(c) for c in df.columns]).show()


# 3. DATA PREPROCESSING
# quit Name, PassengerID and Ticket (identifiers, Not attributes) 
df = df.select('Age', 'Fare', 'Sex', 'Pclass', 'Embarked', 'SibSp', 'Parch', 'Survived', 'Cabin')

# Dealing with missing values
# CABIN: Quit Cabin. The 77.1% of the values are null.
df = df.select('Age', 'Fare', 'Sex', 'Pclass', 'Embarked', 'SibSp', 'Parch', 'Survived')

# AGE: Substitute by Mean
from pyspark.sql.functions import mean as _mean, stddev as _stddev, col
df_stats = df.select(
_mean(col('Age')).alias('mean'),
_stddev(col('Age')).alias('std')).collect()
mean = df_stats[0]['mean']
std = df_stats[0]['std']
# mean=29.7=30
df = df.fillna(30, subset=['Age'])

# FARE: Substitute by Mean
df_stats = df.select(
_mean(col('Fare')).alias('mean')).collect()
mean = df_stats[0]['mean']
# mean=33.2955
df = df.fillna(33.2955, subset=['Fare'])

# EMBARKED: Substitute by most common class
df.groupBy(df['Embarked']).count().show()
# The most common class is 'S'
df = df.fillna('S', subset=['Embarked'])

# See there are no missing values left
df.select([count(when(col(c).isNull(),c)).alias(c) for c in df.columns]).show()

# Create a new column 'relatives' with takesinto account SibSp and parch
df = df.withColumn('Relatives', df.SibSp + df.Parch)

# Prepare features for MLlib
cols = df.columns
df.printSchema()

# convert 'Pclass', 'SibSp', 'Parch' and 'Relatives' to categorical
from pyspark.sql.types import StringType
df = df.withColumn("Pclass", df["Pclass"].cast(StringType()))
df = df.withColumn("SibSp", df["SibSp"].cast(StringType()))
df = df.withColumn("Parch", df["Parch"].cast(StringType()))
df = df.withColumn("Relatives", df["Relatives"].cast(StringType()))

from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
# Categorical features 
categoricalColumns = ['Sex', 'Embarked', 'Pclass','SibSp', 'Parch', 'Relatives']
stages = []

for categoricalCol in categoricalColumns:
	stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
	encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
	stages += [stringIndexer, encoder]
 

# Numeric features
numericCols = ['Age', 'Fare']
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

# Pipeline
from pyspark.ml import Pipeline
pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(df)
df = pipelineModel.transform(df)
selectedCols = ['features'] + cols
df = df.select(selectedCols)
df = df.selectExpr("features", "Survived as label", "Age", "Fare", "Sex", "Pclass", "Embarked", "SibSp", "Parch", "Relatives", "Survived")
df.printSchema()
df.show(5)



# 4. Machine Learning Models

# We shoud have now a label (Survived) and a features columns
# We split in train and test
train, test = df.randomSplit([0.8, 0.2], seed = 2018)
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))

# Logistic Regression Model
print('Logistic Regression Model')
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)
lrModel = lr.fit(train)

# Coefficients
import numpy as np
beta = np.sort(lrModel.coefficients)
print(beta)

# Summarize model over training set
trainingSummary = lrModel.summary

# Roc and Area under the curve
roc = trainingSummary.roc
print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))

# Precision and Recall
pr = trainingSummary.pr
pr.show()

# Predictions on test set
predictions = lrModel.transform(test)
predictions.select('label', 'rawPrediction', 'prediction', 'probability').show(10)

# Evaluate Logistic Regression Model
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC', evaluator.evaluate(predictions))

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
print('Accuracy', evaluator.evaluate(predictions))



print('Decision Tree Classifier')
from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth = 3)
dtModel = dt.fit(train)
predictions = dtModel.transform(test)
predictions.select('label', 'rawPrediction', 'prediction', 'probability').show(10)

# Evaluate Decision Tree
evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
print('Accuracy', evaluator.evaluate(predictions))



print('Random Forest Classifier')
from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
rfModel = rf.fit(train)
predictions = rfModel.transform(test)
predictions.select('label', 'rawPrediction', 'prediction', 'probability').show(10)

# Evaluate Random Forest Classifier
evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
print('Accuracy', evaluator.evaluate(predictions))



print('Gradient-Boosted Tree Classifier')
from pyspark.ml.classification import GBTClassifier
gbt = GBTClassifier(maxIter=10)
gbtModel = gbt.fit(train)
predictions = gbtModel.transform(test)
predictions.select('label', 'rawPrediction', 'prediction', 'probability').show(10)

# Evaluate Gradient-Boosted Tree Classifier
evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
print('Accuracy', evaluator.evaluate(predictions))
# print(gbt.explainParams())

