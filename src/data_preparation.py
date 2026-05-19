import re
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType

# Initialize Spark
spark = SparkSession.builder \
    .appName("ArabicAITextDetection") \
    .getOrCreate()

# Load dataset
spark_df = spark.read.option("header", True) \
    .option("multiline", True) \
    .option("quote", '"') \
    .option("escape", '"') \
    .csv("data/raw/data.csv")
# Normalize Arabic characters
def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub("ة", "ه", text)
    return text

# Remove diacritics
def remove_diacritics(text):
    arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                                 | # Sukun
                             ـ
                         """, re.VERBOSE)
    return re.sub(arabic_diacritics, '', text)

# Text cleaning
def clean_text(text):
    text = remove_diacritics(text)
    text = normalize_arabic(text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    return text

# Arabic stopwords
arabic_stopwords = set([
    "في", "من", "على", "و", "الى", "عن", "هذا", "هذه",
    "ذلك", "كما", "مع", "كان", "قد"
])
# Remove stopwords
def remove_stopwords(text):
    words = text.split()
    filtered = [w for w in words if w not in arabic_stopwords]
    return ' '.join(filtered)

# Simple stemming
def simple_stem(text):
    words = text.split()
    stemmed = [w[:4] for w in words]
    return ' '.join(stemmed)

# Create UDFs
clean_udf = udf(lambda x: clean_text(x) if x else "", StringType())
stop_udf = udf(lambda x: remove_stopwords(x), StringType())
stem_udf = udf(lambda x: simple_stem(x), StringType())
# Apply preprocessing
spark_df = spark_df.withColumn("clean_text", clean_udf(col("text")))
spark_df = spark_df.withColumn("clean_text", stop_udf(col("clean_text")))
spark_df = spark_df.withColumn("processed_text", stem_udf(col("clean_text")))

# Save processed parquet
spark_df.write.mode("overwrite").parquet("data/processed/processed_phase2.parquet")

print("Data preprocessing completed successfully")