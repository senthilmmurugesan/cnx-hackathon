# Databricks notebook source
# /Volumes/skm/demo/data

dbutils.fs.ls("/Volumes/skm/demo/data")

# COMMAND ----------

df = spark.read.csv("/Volumes/skm/demo/data/DEMO*.txt", header=True, sep="$")
df.count()
df.write.mode("overwrite").saveAsTable("skm.demo.demographics")

# COMMAND ----------

df = spark.read.csv("/Volumes/skm/demo/data/DRUG*.txt", header=True, sep="$")
display(df)
df.write.mode("overwrite").saveAsTable("skm.demo.drug")

# COMMAND ----------

df = spark.read.csv("/Volumes/skm/demo/data/REAC*.txt", header=True, sep="$")
display(df)
df.write.mode("overwrite").saveAsTable("skm.demo.reac")

# COMMAND ----------

df = spark.read.csv("/Volumes/skm/demo/data/OUTC*.txt", header=True, sep="$")
display(df)
df.write.mode("overwrite").saveAsTable("skm.demo.outcome")

# COMMAND ----------

