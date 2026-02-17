package com.corc.nah.quality

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

/**
 * Data quality framework for CORC-NAH using Scala traits and functional programming.
 *
 * Demonstrates:
 * - Trait system (similar to interfaces with implementation)
 * - Higher-order functions (map, filter, foreach)
 * - Functional composition
 * - Immutable collections
 */

/**
 * Base trait for quality checks.
 * Traits are like interfaces but can have concrete implementations.
 */
trait QualityCheck {
  def name: String
  def check(df: DataFrame): Boolean
  def metric(df: DataFrame): Double

  // Concrete method in trait (default implementation)
  def report(df: DataFrame): String = {
    val passed = check(df)
    val score = metric(df)
    val status = if (passed) "✅ PASSED" else "❌ FAILED"
    f"$status%-12s $name%-40s (${score * 100}%.2f%%)"
  }
}

/**
 * Check for null values in a column.
 *
 * Demonstrates case class implementing a trait.
 */
case class NotNullCheck(columnName: String, threshold: Double = 1.0) extends QualityCheck {
  override def name: String = s"NotNull: $columnName"

  override def check(df: DataFrame): Boolean = {
    metric(df) >= threshold
  }

  override def metric(df: DataFrame): Double = {
    val total = df.count()
    if (total == 0) return 1.0

    val nonNulls = df.filter(col(columnName).isNotNull).count()
    nonNulls.toDouble / total
  }
}

/**
 * Check that column values match a regex pattern.
 */
case class RegexCheck(
    columnName: String,
    pattern: String,
    threshold: Double = 0.95
) extends QualityCheck {
  override def name: String = s"Regex: $columnName"

  override def check(df: DataFrame): Boolean = {
    metric(df) >= threshold
  }

  override def metric(df: DataFrame): Double = {
    val total = df.count()
    if (total == 0) return 1.0

    val matching = df
      .filter(col(columnName).rlike(pattern))
      .count()

    matching.toDouble / total
  }
}

/**
 * Check for duplicate rows based on specified columns.
 */
case class UniquenessCheck(columns: Seq[String], threshold: Double = 1.0) extends QualityCheck {
  override def name: String = s"Uniqueness: ${columns.mkString(", ")}"

  override def check(df: DataFrame): Boolean = {
    metric(df) >= threshold
  }

  override def metric(df: DataFrame): Double = {
    val total = df.count()
    if (total == 0) return 1.0

    val unique = df.dropDuplicates(columns: _*).count()
    unique.toDouble / total
  }
}

/**
 * Check for minimum row count.
 */
case class RowCountCheck(minRows: Long) extends QualityCheck {
  override def name: String = s"Min Row Count: $minRows"

  override def check(df: DataFrame): Boolean = {
    df.count() >= minRows
  }

  override def metric(df: DataFrame): Double = {
    val count = df.count()
    if (count >= minRows) 1.0 else count.toDouble / minRows
  }
}

/**
 * Main object for running quality checks.
 *
 * Demonstrates functional programming:
 * - Higher-order functions (map, filter)
 * - Immutable collections
 * - Functional composition
 */
object DataQuality {

  def main(args: Array[String]): Unit = {
    val inputPath = if (args.length > 0) args(0) else "data/silver/corpus.parquet"

    val spark = SparkSession.builder()
      .appName("CORC-NAH Data Quality")
      .master("local[*]")
      .getOrCreate()

    try {
      // Load data
      println(s"Loading data from: $inputPath")
      val df = spark.read.parquet(inputPath)

      // Define quality checks (immutable list)
      val checks: List[QualityCheck] = List(
        NotNullCheck("nah_text"),
        NotNullCheck("spanish_text"),
        RegexCheck("nah_text", "^[\\p{L}\\s.,;!?áéíóúāēīōū]+$", threshold = 0.95),
        RegexCheck("spanish_text", "^[\\p{L}\\s.,;!?áéíóúñÑ¿¡]+$", threshold = 0.95),
        UniquenessCheck(Seq("nah_text", "spanish_text"), threshold = 0.98),
        RowCountCheck(minRows = 1000)
      )

      println("\n" + "=" * 80)
      println("Data Quality Report - CORC-NAH Corpus")
      println("=" * 80)
      println(f"Total Records: ${df.count()}")
      println("-" * 80)

      // Run checks using functional programming
      val results = checks.map { check =>
        (check, check.check(df), check.metric(df))
      }

      // Print results (foreach is a higher-order function)
      results.foreach { case (check, passed, _) =>
        println(check.report(df))
      }

      println("-" * 80)

      // Summary statistics (functional: filter + count)
      val totalChecks = results.length
      val passedChecks = results.count { case (_, passed, _) => passed }
      val failedChecks = totalChecks - passedChecks

      println(f"\nSummary:")
      println(f"  Total Checks:  $totalChecks%-3d")
      println(f"  Passed:        $passedChecks%-3d (${passedChecks.toDouble / totalChecks * 100}%.1f%%)")
      println(f"  Failed:        $failedChecks%-3d")

      // Exit code based on results (functional: exists)
      val anyFailed = results.exists { case (_, passed, _) => !passed }
      if (anyFailed) {
        println("\n❌ Quality checks FAILED")
        sys.exit(1)
      } else {
        println("\n✅ All quality checks PASSED")
      }

    } finally {
      spark.stop()
    }
  }

  /**
   * Run quality checks and return results.
   *
   * Demonstrates functional approach: pure function, no side effects.
   */
  def runChecks(df: DataFrame, checks: List[QualityCheck]): List[(QualityCheck, Boolean, Double)] = {
    checks.map { check =>
      (check, check.check(df), check.metric(df))
    }
  }

  /**
   * Filter checks by result (pass/fail).
   *
   * Demonstrates higher-order functions as parameters.
   */
  def filterResults(
      results: List[(QualityCheck, Boolean, Double)],
      predicate: Boolean => Boolean
  ): List[(QualityCheck, Boolean, Double)] = {
    results.filter { case (_, passed, _) => predicate(passed) }
  }
}
