package com.corc.nah.pipeline

import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{MinHashLSH, HashingTF}
import org.apache.spark.ml.linalg.Vector

/**
 * Fuzzy deduplication for the CORC-NAH corpus using MinHash LSH.
 *
 * This module demonstrates core Scala/Spark concepts:
 * - Functional programming: immutability, transformations
 * - Case classes for structured data
 * - Pattern matching
 * - Option handling (avoiding null)
 */
object SparkDedup {

  // Case class for bilingual pairs (immutable by default)
  case class BilingualPair(
      id: String,
      nahText: String,
      spanishText: String,
      source: String
  )

  def main(args: Array[String]): Unit = {
    // Parse command line arguments
    val config = parseArgs(args)

    // Initialize Spark session with optimizations
    val spark = createSparkSession()
    import spark.implicits._

    try {
      // 1. Load data from Silver layer
      println(s"Loading data from: ${config.inputPath}")
      val df = spark.read.parquet(config.inputPath)

      // 2. Exact deduplication (functional transformation)
      val exactDeduped = df
        .dropDuplicates("nah_text", "spanish_text")
        .cache() // Performance optimization

      val initialCount = df.count()
      val dedupedCount = exactDeduped.count()
      println(s"Records before exact dedup: $initialCount")
      println(s"Records after exact dedup: $dedupedCount")
      println(f"Duplicate rate: ${(1 - dedupedCount.toDouble / initialCount) * 100}%.2f%%")

      // 3. Fuzzy deduplication with MinHash LSH
      val fuzzyResult = performFuzzyDedup(
        exactDeduped,
        threshold = config.threshold
      )

      // 4. Pattern matching on Option[DataFrame]
      fuzzyResult match {
        case Some(duplicates) if duplicates.count() > 0 =>
          println(s"Fuzzy duplicates found: ${duplicates.count()}")

          // Remove duplicates (keep first occurrence)
          val duplicateIds = duplicates
            .select("datasetB.id")
            .distinct()
            .as[String]
            .collect()
            .toSet

          val finalDeduped = exactDeduped
            .filter(!col("id").isin(duplicateIds.toSeq: _*))

          println(s"Final record count: ${finalDeduped.count()}")

          // Write to Diamond layer
          finalDeduped.write
            .mode("overwrite")
            .parquet(config.outputPath)

          println(s"✅ Deduplication complete. Output: ${config.outputPath}")

        case Some(duplicates) =>
          println("No fuzzy duplicates found")
          exactDeduped.write.mode("overwrite").parquet(config.outputPath)

        case None =>
          println("Fuzzy dedup skipped (dataset too small)")
          exactDeduped.write.mode("overwrite").parquet(config.outputPath)
      }

    } finally {
      spark.stop()
    }
  }

  /**
   * Perform fuzzy deduplication using MinHash LSH algorithm.
   *
   * Demonstrates:
   * - Option type for null safety
   * - Functional pipeline (no side effects)
   * - Immutable transformations
   *
   * @param df Input DataFrame
   * @param threshold Similarity threshold (0.0-1.0)
   * @return Option[DataFrame] with duplicate pairs, or None if dataset too small
   */
  def performFuzzyDedup(
      df: DataFrame,
      threshold: Double = 0.8
  ): Option[DataFrame] = {

    if (df.isEmpty) {
      return None
    }

    // Create combined text column
    val withCombined = df.withColumn(
      "combined_text",
      concat_ws(" ", col("nah_text"), col("spanish_text"))
    )

    // Tokenize and create features using HashingTF
    val tokenized = withCombined.withColumn(
      "words",
      split(col("combined_text"), "\\s+")
    )

    val hashingTF = new HashingTF()
      .setInputCol("words")
      .setOutputCol("features")
      .setNumFeatures(1000)

    val featurized = hashingTF.transform(tokenized)

    // MinHash LSH model
    val mh = new MinHashLSH()
      .setNumHashTables(5)
      .setInputCol("features")
      .setOutputCol("hashes")

    val model = mh.fit(featurized)

    // Self-join to find similar pairs
    val duplicates = model
      .approxSimilarityJoin(featurized, featurized, threshold, "distance")
      .filter(col("datasetA.id") =!= col("datasetB.id")) // Exclude self-matches
      .filter(col("datasetA.id") < col("datasetB.id")) // Avoid duplicate pairs

    Some(duplicates)
  }

  /**
   * Create Spark session with optimized settings for ETL workloads.
   */
  private def createSparkSession(): SparkSession = {
    SparkSession.builder()
      .appName("CORC-NAH Deduplication")
      .config("spark.driver.memory", "4g")
      .config("spark.executor.memory", "8g")
      .config("spark.sql.shuffle.partitions", "200")
      .config("spark.sql.adaptive.enabled", "true")
      .master("local[*]") // For local execution; remove for cluster
      .getOrCreate()
  }

  /**
   * Parse command line arguments with default values.
   *
   * Demonstrates case class for configuration.
   */
  private def parseArgs(args: Array[String]): Config = {
    val inputPath = if (args.length > 0) args(0) else "data/silver/corpus.parquet"
    val outputPath = if (args.length > 1) args(1) else "data/diamond/deduped.parquet"
    val threshold = if (args.length > 2) args(2).toDouble else 0.8

    Config(inputPath, outputPath, threshold)
  }

  case class Config(
      inputPath: String,
      outputPath: String,
      threshold: Double
  )
}
