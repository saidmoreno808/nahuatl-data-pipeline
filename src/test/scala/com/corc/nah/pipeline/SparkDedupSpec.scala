package com.corc.nah.pipeline

import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll

/**
 * Tests for SparkDedup module.
 *
 * Demonstrates Scala testing with ScalaTest and Spark.
 */
class SparkDedupSpec extends AnyFunSuite with BeforeAndAfterAll {

  var spark: SparkSession = _

  override def beforeAll(): Unit = {
    spark = SparkSession.builder()
      .appName("SparkDedup Tests")
      .master("local[2]")
      .config("spark.ui.enabled", "false")
      .getOrCreate()
  }

  override def afterAll(): Unit = {
    if (spark != null) {
      spark.stop()
    }
  }

  test("performFuzzyDedup should return None for empty DataFrame") {
    import spark.implicits._

    val emptyDF = Seq.empty[(String, String, String, String)]
      .toDF("id", "nah_text", "spanish_text", "source")

    val result = SparkDedup.performFuzzyDedup(emptyDF)

    assert(result.isEmpty, "Should return None for empty DataFrame")
  }

  test("performFuzzyDedup should find similar text pairs") {
    import spark.implicits._

    val testData = Seq(
      ("1", "Niltze", "Hola", "test"),
      ("2", "Niltze", "Hola", "test"),  // Exact duplicate
      ("3", "Tlazohcamati", "Gracias", "test"),
      ("4", "Tlazohcāmati", "Gracias", "test")  // Macron variant
    ).toDF("id", "nah_text", "spanish_text", "source")

    val result = SparkDedup.performFuzzyDedup(testData, threshold = 0.9)

    assert(result.isDefined, "Should return Some for non-empty DataFrame")
    val duplicates = result.get
    assert(duplicates.count() > 0, "Should find at least one duplicate pair")
  }

  test("Config case class should parse arguments correctly") {
    val args = Array("input.parquet", "output.parquet", "0.75")
    val config = SparkDedup.parseArgs(args)

    assert(config.inputPath == "input.parquet")
    assert(config.outputPath == "output.parquet")
    assert(config.threshold == 0.75)
  }

  test("Config should use defaults when no args provided") {
    val config = SparkDedup.parseArgs(Array())

    assert(config.inputPath == "data/silver/corpus.parquet")
    assert(config.outputPath == "data/diamond/deduped.parquet")
    assert(config.threshold == 0.8)
  }
}
