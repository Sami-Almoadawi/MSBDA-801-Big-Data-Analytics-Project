from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Load processed data
spark_df = spark.read.parquet("data/processed/processed_phase2.parquet")

# Feature 6
from pyspark.sql.functions import udf

def elongation_count(text):
    return text.count("اا")
elong_udf = udf(lambda x: elongation_count(x), IntegerType())

# Feature 29

def semicolon_count(text):
    return text.count("؛")

semi_udf = udf(lambda x: semicolon_count(x), IntegerType())

# Feature 52
interjections = ["واو", "آه", "يا", "هه"]

def count_interjections(text):
    words = text.split()
    return sum(1 for w in words if w in interjections)

interj_udf = udf(lambda x: count_interjections(x), IntegerType())

# Feature 75
def active_voice(text):
    return text.count("ي")

active_udf = udf(lambda x: active_voice(x), IntegerType())

# Feature 98

def bert_representation(text):
    return len(text)

bert_udf = udf(lambda x: bert_representation(x), IntegerType())

# Add features
spark_df = spark_df \
    .withColumn("feat_6_elong", elong_udf(col("text"))) \
    .withColumn("feat_29_semicolon", semi_udf(col("text"))) \
    .withColumn("feat_52_interj", interj_udf(col("text"))) \
    .withColumn("feat_75_active", active_udf(col("text"))) \
    .withColumn("feat_98_bert", bert_udf(col("text")))
# TF-IDF Pipeline
tokenizer = Tokenizer(inputCol="processed_text", outputCol="words")
words_data = tokenizer.transform(spark_df)

remover = StopWordsRemover(inputCol="words", outputCol="filtered")
filtered_data = remover.transform(words_data)

hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
featurized_data = hashingTF.transform(filtered_data)

idf = IDF(inputCol="rawFeatures", outputCol="tfidf_features")
idf_model = idf.fit(featurized_data)
df_tfidf = idf_model.transform(featurized_data)

# Assemble features
assembler = VectorAssembler(
    inputCols=[
            "tfidf_features",
        "feat_6_elong",
        "feat_29_semicolon",
        "feat_52_interj",
        "feat_75_active",
        "feat_98_bert"
    ],
    outputCol="features"
)

final_df = assembler.transform(df_tfidf)

# Label casting
final_df = final_df.withColumn("label", col("label").cast("int"))

# Split data
train_df, temp_df = final_df.randomSplit([0.7, 0.3], seed=42)
val_df, test_df = temp_df.randomSplit([0.5, 0.5], seed=42)
# Logistic Regression
lr = LogisticRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(train_df)
lr_predictions = lr_model.transform(test_df)

# Random Forest
rf = RandomForestClassifier(featuresCol="features", labelCol="label")
rf_model = rf.fit(train_df)
rf_predictions = rf_model.transform(test_df)

# SVM
svm = LinearSVC(featuresCol="features", labelCol="label")
svm_model = svm.fit(train_df)
svm_predictions = svm_model.transform(test_df)

# Evaluation
evaluator = MulticlassClassificationEvaluator(labelCol="label")
print("LR Accuracy:", evaluator.evaluate(lr_predictions, {evaluator.metricName: "accuracy"}))
print("RF Accuracy:", evaluator.evaluate(rf_predictions, {evaluator.metricName: "accuracy"}))
print("SVM Accuracy:", evaluator.evaluate(svm_predictions, {evaluator.metricName: "accuracy"}))

# ROC-AUC
evaluator_auc = BinaryClassificationEvaluator(labelCol="label")

print("LR AUC:", evaluator_auc.evaluate(lr_predictions))
print("RF AUC:", evaluator_auc.evaluate(rf_predictions))
print("SVM AUC:", evaluator_auc.evaluate(svm_predictions))