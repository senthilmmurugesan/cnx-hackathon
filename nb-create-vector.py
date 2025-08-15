# Databricks notebook source
# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE skm.demo.curated_drug_names AS
# MAGIC SELECT 
# MAGIC     monotonically_increasing_id() AS id,
# MAGIC     LOWER(TRIM(drugname)) AS drug_name
# MAGIC FROM (
# MAGIC     SELECT DISTINCT drugname
# MAGIC     FROM skm.demo.drug
# MAGIC );

# COMMAND ----------

