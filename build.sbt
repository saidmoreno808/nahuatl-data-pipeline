name := "corc-nah-spark"

version := "1.0.0"

scalaVersion := "2.12.18"

// Spark dependencies
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.5.0" % "provided",
  "org.apache.spark" %% "spark-sql" % "3.5.0" % "provided",
  "org.apache.spark" %% "spark-mllib" % "3.5.0" % "provided",
  
  // Testing
  "org.scalatest" %% "scalatest" % "3.2.17" % Test
)

// JVM settings for  Spark
javaOptions ++= Seq(
  "-Xms512M",
  "-Xmx2048M",
  "-XX:+CMSClassUnloadingEnabled"
)

// Assembly settings for fat JAR
assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}

// Test settings
testOptions in Test += Tests.Argument("-oD")
parallelExecution in Test := false
