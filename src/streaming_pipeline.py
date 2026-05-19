import os
from pyspark.sql.functions import col

# Create streaming folder
os.makedirs("stream_data", exist_ok=True)

# Read stream
stream_df = spark.readStream \
    .format("text") \
    .option("maxFilesPerTrigger", 1) \
    .load("stream_data")

# Rename column
stream_df = stream_df.withColumnRenamed("value", "text")

# Apply preprocessing
stream_df = stream_df.withColumn("clean_text", clean_udf(col("text")))
stream_df = stream_df.withColumn("processed_text", stem_udf(col("clean_text")))

# TF-IDF
stream_tokens = tokenizer.transform(stream_df)
stream_filtered = remover.transform(stream_tokens)
stream_tf = hashingTF.transform(stream_filtered)
stream_tfidf = idf_model.transform(stream_tf)

# Add features
stream_features = stream_tfidf \
    .withColumn("feat_6_elong", elong_udf(col("text"))) \
    .withColumn("feat_29_semicolon", semi_udf(col("text"))) \
    .withColumn("feat_52_interj", interj_udf(col("text"))) \
    .withColumn("feat_75_active", active_udf(col("text"))) \
    .withColumn("feat_98_bert", bert_udf(col("text")))

# Assemble features
stream_final = assembler.transform(stream_features)

# Predictions
predictions = svm_model.transform(stream_final)

# Streaming output
query = predictions.select("text", "prediction") \
    .writeStream \
    .format("console") \
    .option("truncate", False) \
    .start()

query.awaitTermination()