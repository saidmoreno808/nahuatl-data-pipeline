// CORC-NAH ETL Pipeline - Declarative Jenkinsfile
// Demonstrates understanding of CI/CD orchestration

pipeline {
    agent {
        docker {
            image 'python:3.10-slim'
            args '-v $WORKSPACE:/workspace -w /workspace'
        }
    }

    options {
        // Build timeout
        timeout(time: 1, unit: 'HOURS')

        // Keep only last 10 builds
        buildDiscarder(logRotator(numToKeepStr: '10'))

        // Disable concurrent builds
        disableConcurrentBuilds()

        // Timestamps in console output
        timestamps()

        // ANSI colors in console
        ansiColor('xterm')
    }

    environment {
        // Python environment
        PYTHONUNBUFFERED = '1'
        PYTHONDONTWRITEBYTECODE = '1'

        // UTF-8 encoding
        LANG = 'en_US.UTF-8'
        LC_ALL = 'en_US.UTF-8'

        // Virtual environment
        VENV_DIR = '.venv'

        // Git configuration
        GIT_COMMIT_SHORT = sh(
            script: "git rev-parse --short HEAD",
            returnStdout: true
        ).trim()
    }

    stages {
        stage('Setup') {
            steps {
                echo '📦 Installing dependencies...'

                sh '''
                    python3 -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate

                    pip install --upgrade pip setuptools wheel
                    pip install -r requirements.txt
                    pip install -r requirements-dev.txt
                '''
            }
        }

        stage('Code Quality') {
            parallel {
                stage('Linting') {
                    steps {
                        echo '🔍 Running linters...'

                        sh '''
                            . ${VENV_DIR}/bin/activate

                            # Flake8
                            flake8 src/ tests/ --max-line-length=100 \
                                --exclude=__pycache__,.venv \
                                --format=pylint > flake8-report.txt || true

                            # Pylint
                            pylint src/ --output-format=parseable \
                                --reports=no > pylint-report.txt || true
                        '''

                        // Archive reports
                        archiveArtifacts artifacts: '*-report.txt', allowEmptyArchive: true
                    }
                }

                stage('Type Checking') {
                    steps {
                        echo '🔍 Running mypy...'

                        sh '''
                            . ${VENV_DIR}/bin/activate

                            mypy src/ --ignore-missing-imports \
                                --no-error-summary > mypy-report.txt || true
                        '''

                        archiveArtifacts artifacts: 'mypy-report.txt', allowEmptyArchive: true
                    }
                }

                stage('Formatting Check') {
                    steps {
                        echo '🎨 Checking code formatting...'

                        sh '''
                            . ${VENV_DIR}/bin/activate

                            # Check if code is formatted
                            black --check src/ tests/ || exit 1

                            # Check import sorting
                            isort --check-only src/ tests/ --profile black || exit 1
                        '''
                    }
                }
            }
        }

        stage('Unit Tests') {
            steps {
                echo '🧪 Running unit tests...'

                sh '''
                    . ${VENV_DIR}/bin/activate

                    pytest tests/unit/ \
                        -v \
                        --junitxml=junit-unit.xml \
                        --cov=src \
                        --cov-report=xml:coverage-unit.xml \
                        --cov-report=html:htmlcov-unit
                '''

                // Publish test results
                junit 'junit-unit.xml'

                // Publish coverage
                publishHTML([
                    reportDir: 'htmlcov-unit',
                    reportFiles: 'index.html',
                    reportName: 'Unit Test Coverage'
                ])
            }
        }

        stage('Parity Tests') {
            when {
                // Only run on main branch or PRs
                anyOf {
                    branch 'main'
                    changeRequest()
                }
            }

            steps {
                echo '🔍 Running parity tests with legacy pipeline...'

                sh '''
                    . ${VENV_DIR}/bin/activate

                    # Generate golden dataset if not exists
                    if [ ! -f "benchmark/golden_train_v1.jsonl" ]; then
                        echo "⚠️  Golden dataset not found. Generating..."
                        make golden
                    fi

                    # Run parity tests
                    pytest tests/integration/test_parity_with_legacy.py \
                        -v \
                        --junitxml=junit-parity.xml \
                        --html=parity-report.html \
                        --self-contained-html
                '''

                // Fail build if parity tests don't pass
                junit 'junit-parity.xml'

                archiveArtifacts artifacts: 'parity-report.html', allowEmptyArchive: true

                script {
                    def testResults = junit 'junit-parity.xml'

                    if (testResults.failCount > 0) {
                        error("❌ Parity tests failed! Refactored code doesn't match legacy behavior.")
                    }
                }
            }
        }

        stage('Integration Tests') {
            steps {
                echo '🧪 Running integration tests...'

                sh '''
                    . ${VENV_DIR}/bin/activate

                    pytest tests/integration/ \
                        -v \
                        --junitxml=junit-integration.xml \
                        --cov=src \
                        --cov-append \
                        --cov-report=xml:coverage-integration.xml
                '''

                junit 'junit-integration.xml'
            }
        }

        stage('Data Quality Validation') {
            when {
                branch 'main'
            }

            steps {
                echo '📊 Running data quality checks...'

                sh '''
                    . ${VENV_DIR}/bin/activate

                    # Run Great Expectations validation
                    # (Commented out until GE suite is configured)
                    # great_expectations checkpoint run gold_dataset_validation

                    # Generate quality report
                    python benchmark/generate_stats.py
                '''

                archiveArtifacts artifacts: 'benchmark/golden_stats.json', allowEmptyArchive: true
            }
        }

        stage('Build Documentation') {
            steps {
                echo '📚 Building documentation...'

                sh '''
                    . ${VENV_DIR}/bin/activate

                    # Generate API docs (if sphinx configured)
                    # cd docs && make html

                    echo "Documentation build placeholder"
                '''
            }
        }

        stage('Package') {
            when {
                anyOf {
                    branch 'main'
                    tag 'v*'
                }
            }

            steps {
                echo '📦 Creating distribution package...'

                sh '''
                    . ${VENV_DIR}/bin/activate

                    # Build wheel
                    python setup.py sdist bdist_wheel

                    # Generate SHA256 checksums
                    cd dist && sha256sum * > checksums.txt
                '''

                archiveArtifacts artifacts: 'dist/*', allowEmptyArchive: true
            }
        }

        stage('Deploy') {
            when {
                tag 'v*'
            }

            steps {
                echo '🚀 Deploying to production...'

                script {
                    // Placeholder for deployment logic
                    echo "Would deploy version ${env.TAG_NAME}"
                    echo "Commit: ${env.GIT_COMMIT_SHORT}"

                    // In real scenario:
                    // - Push Docker image to registry
                    // - Update Control-M job definitions
                    // - Deploy to AWS Lambda/EC2
                    // - Update data catalog in AWS Glue
                }
            }
        }
    }

    post {
        always {
            echo '🧹 Cleaning up...'

            // Clean workspace
            cleanWs(
                deleteDirs: true,
                patterns: [
                    [pattern: '.venv/**', type: 'INCLUDE'],
                    [pattern: '__pycache__/**', type: 'INCLUDE'],
                    [pattern: '*.pyc', type: 'INCLUDE']
                ]
            )
        }

        success {
            echo '✅ Pipeline completed successfully!'

            // Send notification (Slack, email, etc.)
            script {
                def message = """
                    ✅ Build #${env.BUILD_NUMBER} succeeded
                    Branch: ${env.BRANCH_NAME}
                    Commit: ${env.GIT_COMMIT_SHORT}
                    Duration: ${currentBuild.durationString}
                """.stripIndent()

                echo message

                // In real scenario:
                // slackSend(channel: '#data-engineering', message: message)
            }
        }

        failure {
            echo '❌ Pipeline failed!'

            // Send failure notification
            script {
                def message = """
                    ❌ Build #${env.BUILD_NUMBER} failed
                    Branch: ${env.BRANCH_NAME}
                    Commit: ${env.GIT_COMMIT_SHORT}
                    Stage: ${env.STAGE_NAME}
                """.stripIndent()

                echo message

                // In real scenario:
                // slackSend(channel: '#data-engineering', color: 'danger', message: message)
            }
        }

        unstable {
            echo '⚠️  Pipeline completed with warnings'
        }
    }
}

// ============================================================================
// Control-M Integration Notes
// ============================================================================

/*
In a real enterprise environment with Control-M, this Jenkins job would be
triggered by Control-M using:

1. Control-M Job Definition:
   - Job Type: "Jenkins Job"
   - Jenkins URL: https://jenkins.company.com
   - Job Path: corc-nah/main
   - Parameters: { "ENVIRONMENT": "production" }

2. Scheduling:
   - Daily at 02:00 UTC
   - Wait for upstream data ingestion jobs
   - On-condition: File watcher for new data in S3

3. Dependencies:
   - WAIT: Job "S3_DATA_SYNC" (ensures new data is available)
   - WAIT: Job "DB_BACKUP" (ensures metadata DB is backed up)
   - RUNS: This Jenkins pipeline
   - NEXT: Job "DATA_CATALOG_UPDATE" (updates AWS Glue catalog)

4. Notifications:
   - On-Success: Email to data-engineering@company.com
   - On-Failure: Page on-call engineer + create PagerDuty incident
   - On-Late: Email warning if not completed by 04:00 UTC

5. Resource Management:
   - Resource Pool: "DATA_PROCESSING_POOL" (limit concurrent ETL jobs)
   - Max Runtime: 2 hours
   - Auto-rerun: 1 time with 5-minute delay

See docs/controlm-integration.md for full integration guide.
*/
