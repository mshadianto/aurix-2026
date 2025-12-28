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
]
