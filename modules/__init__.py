"""
AURIX 2026 Excellence Modules.
New intelligent modules for agentic audit platform.
"""

from modules.process_mining import (
    # Core Process Mining
    generate_sample_event_log,
    parse_event_log,
    calculate_dfg,
    calculate_activity_durations,
    detect_bottlenecks,
    get_process_variants,
    generate_dfg_graphviz,
    calculate_process_metrics,
    BottleneckInfo,
    # Conformance Twin
    ConformanceStatus,
    ActivityDeviation,
    SOPActivity,
    SOPSchema,
    ConformanceResult,
    ConformanceSummary,
    LOAN_APPROVAL_SOP,
    load_sop_schema,
    check_case_conformance,
    conformance_twin_analysis,
    export_sop_to_json,
)

from modules.regulatory_rag import (
    RegulatoryValidator,
    ComplianceResult,
    ComplianceStatus,
    ESGCategory,
    RegulatoryReference,
    REGULATORY_KNOWLEDGE_BASE,
    SAMPLE_QUERIES
)

from modules.anti_fraud_agent import (
    # Enums
    RiskLevel,
    AlertType,
    # Models
    TransactionRecord,
    CustomerProfile,
    AlertEvidence,
    FraudAlert,
    SARNarrative,
    FraudAnalysisSummary,
    # Constants
    CASH_THRESHOLD_IDR,
    TRANSFER_THRESHOLD_IDR,
    HIGH_RISK_COUNTRIES,
    # Classes
    AntiFraudAgent,
    # Functions
    generate_sample_transactions,
)

from modules.ijk_benchmarking import (
    # Enums
    InstitutionType,
    MetricCategory,
    BenchmarkStatus,
    # Models
    IndustryBenchmark,
    EntityMetric,
    BenchmarkResult,
    BenchmarkSummary,
    # Constants
    BANKING_BENCHMARKS_2024,
    INSURANCE_BENCHMARKS_2024,
    # Classes
    IJKBenchmarkEngine,
    # Functions
    generate_sample_entity_metrics,
)

from modules.stress_tester import (
    # Enums
    ScenarioSeverity,
    RiskType,
    StressOutcome,
    # Models
    PortfolioPosition,
    StressScenario,
    StressImpact,
    ProjectedMetrics,
    StressTestResult,
    StressTestSuite,
    # Constants
    STRESS_SCENARIOS,
    REGULATORY_THRESHOLDS,
    BI_RATE_BASELINE,
    USDIDR_BASELINE,
    # Classes
    MacroStressTester,
    # Functions
    generate_sample_portfolio,
)

from modules.risk_habit_scorecard import (
    # Enums
    HabitCategory,
    HabitFrequency,
    ScoreLevel,
    TrendDirection,
    # Models
    HabitDefinition,
    HabitExecution,
    HabitStreak,
    UserHabitScore,
    UserScorecard,
    TeamScorecard,
    ComplianceNudge,
    # Constants
    RISK_HABITS,
    # Classes
    RiskHabitEngine,
    # Functions
    generate_sample_executions,
)

__all__ = [
    # Process Mining - Core
    "generate_sample_event_log",
    "parse_event_log",
    "calculate_dfg",
    "calculate_activity_durations",
    "detect_bottlenecks",
    "get_process_variants",
    "generate_dfg_graphviz",
    "calculate_process_metrics",
    "BottleneckInfo",
    # Process Mining - Conformance Twin
    "ConformanceStatus",
    "ActivityDeviation",
    "SOPActivity",
    "SOPSchema",
    "ConformanceResult",
    "ConformanceSummary",
    "LOAN_APPROVAL_SOP",
    "load_sop_schema",
    "check_case_conformance",
    "conformance_twin_analysis",
    "export_sop_to_json",
    # Regulatory RAG
    "RegulatoryValidator",
    "ComplianceResult",
    "ComplianceStatus",
    "ESGCategory",
    "RegulatoryReference",
    "REGULATORY_KNOWLEDGE_BASE",
    "SAMPLE_QUERIES",
    # Anti-Fraud Agent
    "RiskLevel",
    "AlertType",
    "TransactionRecord",
    "CustomerProfile",
    "AlertEvidence",
    "FraudAlert",
    "SARNarrative",
    "FraudAnalysisSummary",
    "CASH_THRESHOLD_IDR",
    "TRANSFER_THRESHOLD_IDR",
    "HIGH_RISK_COUNTRIES",
    "AntiFraudAgent",
    "generate_sample_transactions",
    # IJK Benchmarking
    "InstitutionType",
    "MetricCategory",
    "BenchmarkStatus",
    "IndustryBenchmark",
    "EntityMetric",
    "BenchmarkResult",
    "BenchmarkSummary",
    "BANKING_BENCHMARKS_2024",
    "INSURANCE_BENCHMARKS_2024",
    "IJKBenchmarkEngine",
    "generate_sample_entity_metrics",
    # Macro-Financial Stress Tester
    "ScenarioSeverity",
    "RiskType",
    "StressOutcome",
    "PortfolioPosition",
    "StressScenario",
    "StressImpact",
    "ProjectedMetrics",
    "StressTestResult",
    "StressTestSuite",
    "STRESS_SCENARIOS",
    "REGULATORY_THRESHOLDS",
    "BI_RATE_BASELINE",
    "USDIDR_BASELINE",
    "MacroStressTester",
    "generate_sample_portfolio",
    # Risk Habit Scorecard
    "HabitCategory",
    "HabitFrequency",
    "ScoreLevel",
    "TrendDirection",
    "HabitDefinition",
    "HabitExecution",
    "HabitStreak",
    "UserHabitScore",
    "UserScorecard",
    "TeamScorecard",
    "ComplianceNudge",
    "RISK_HABITS",
    "RiskHabitEngine",
    "generate_sample_executions",
]
