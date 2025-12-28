"""
Process Mining Module for AURIX 2026.
Automated process discovery with Directly-Follows Graph (DFG) visualization.

Features:
- Event log parsing (CSV format)
- DFG calculation and visualization
- Bottleneck detection
- Process variant analysis
- Conformance Twin (SOP comparison)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum
from pydantic import BaseModel, Field
import random
import json


@dataclass
class BottleneckInfo:
    """Information about a detected bottleneck."""
    activity: str
    avg_duration_hours: float
    event_count: int
    severity: str  # "high", "medium", "low"
    percentile_rank: float


def generate_sample_event_log(num_cases: int = 100) -> pd.DataFrame:
    """
    Generate sample event log for loan approval process.
    
    Columns: case_id, activity, timestamp, resource
    """
    activities = [
        ("Application Received", 0.5, 2),      # (name, min_hours, max_hours)
        ("Document Verification", 4, 24),
        ("Credit Check", 2, 8),
        ("Risk Assessment", 24, 72),           # Intentional bottleneck
        ("Manager Approval", 4, 12),
        ("Final Review", 2, 6),
        ("Loan Disbursement", 1, 4)
    ]
    
    resources = ["Ahmad", "Budi", "Citra", "Dewi", "Eko", "Fitri"]
    
    events = []
    base_date = datetime.now() - timedelta(days=90)
    
    for case_num in range(1, num_cases + 1):
        case_id = f"LOAN-2024-{case_num:04d}"
        current_time = base_date + timedelta(days=random.uniform(0, 60))
        
        # Determine if this case has variants
        skip_credit = random.random() < 0.1  # 10% skip credit check
        needs_escalation = random.random() < 0.15  # 15% need escalation
        
        for i, (activity, min_h, max_h) in enumerate(activities):
            # Skip credit check for some cases
            if activity == "Credit Check" and skip_credit:
                continue
            
            # Add event
            events.append({
                "case_id": case_id,
                "activity": activity,
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "resource": random.choice(resources)
            })
            
            # Calculate next timestamp
            duration_hours = random.uniform(min_h, max_h)
            
            # Add extra time for bottleneck (Risk Assessment)
            if activity == "Risk Assessment":
                duration_hours *= random.uniform(1.2, 2.0)
            
            current_time += timedelta(hours=duration_hours)
            
            # Add escalation loop
            if activity == "Manager Approval" and needs_escalation:
                events.append({
                    "case_id": case_id,
                    "activity": "Escalation Review",
                    "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "resource": "Manager"
                })
                current_time += timedelta(hours=random.uniform(8, 24))
    
    df = pd.DataFrame(events)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)


def parse_event_log(df: pd.DataFrame, case_col: str, activity_col: str, timestamp_col: str) -> pd.DataFrame:
    """Parse and validate event log dataframe."""
    required_cols = [case_col, activity_col, timestamp_col]
    
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Standardize column names
    result = df[[case_col, activity_col, timestamp_col]].copy()
    result.columns = ["case_id", "activity", "timestamp"]
    
    # Parse timestamp
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    
    # Sort by case and timestamp
    result = result.sort_values(["case_id", "timestamp"]).reset_index(drop=True)
    
    return result


def calculate_dfg(df: pd.DataFrame) -> Tuple[Dict[Tuple[str, str], int], Dict[str, int]]:
    """
    Calculate Directly-Follows Graph from event log.
    
    Returns:
        dfg_frequencies: Dict mapping (source, target) to frequency
        activity_counts: Dict mapping activity to total count
    """
    dfg = defaultdict(int)
    activity_counts = defaultdict(int)
    
    for case_id, group in df.groupby("case_id"):
        activities = group.sort_values("timestamp")["activity"].tolist()
        
        for activity in activities:
            activity_counts[activity] += 1
        
        for i in range(len(activities) - 1):
            source = activities[i]
            target = activities[i + 1]
            dfg[(source, target)] += 1
    
    return dict(dfg), dict(activity_counts)


def calculate_activity_durations(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate average duration for each activity in hours."""
    durations = defaultdict(list)
    
    for case_id, group in df.groupby("case_id"):
        sorted_group = group.sort_values("timestamp")
        timestamps = sorted_group["timestamp"].tolist()
        activities = sorted_group["activity"].tolist()
        
        for i in range(len(timestamps) - 1):
            activity = activities[i]
            duration = (timestamps[i + 1] - timestamps[i]).total_seconds() / 3600
            durations[activity].append(duration)
    
    # Calculate averages
    avg_durations = {}
    for activity, dur_list in durations.items():
        if dur_list:
            avg_durations[activity] = sum(dur_list) / len(dur_list)
        else:
            avg_durations[activity] = 0
    
    return avg_durations


def detect_bottlenecks(
    df: pd.DataFrame,
    threshold_percentile: float = 75
) -> List[BottleneckInfo]:
    """
    Detect bottleneck activities based on duration.
    
    Args:
        df: Event log dataframe
        threshold_percentile: Percentile threshold for bottleneck detection
    
    Returns:
        List of BottleneckInfo objects
    """
    durations = calculate_activity_durations(df)
    activity_counts = df.groupby("activity").size().to_dict()
    
    if not durations:
        return []
    
    # Calculate threshold
    all_durations = list(durations.values())
    threshold = np.percentile(all_durations, threshold_percentile)
    
    bottlenecks = []
    
    for activity, avg_duration in sorted(durations.items(), key=lambda x: -x[1]):
        # Calculate percentile rank
        rank = sum(1 for d in all_durations if d <= avg_duration) / len(all_durations) * 100
        
        # Determine severity
        if avg_duration >= np.percentile(all_durations, 90):
            severity = "high"
        elif avg_duration >= threshold:
            severity = "medium"
        else:
            severity = "low"
        
        if avg_duration >= threshold:
            bottlenecks.append(BottleneckInfo(
                activity=activity,
                avg_duration_hours=round(avg_duration, 2),
                event_count=activity_counts.get(activity, 0),
                severity=severity,
                percentile_rank=round(rank, 1)
            ))
    
    return bottlenecks


def get_process_variants(df: pd.DataFrame, top_n: int = 5) -> List[Dict]:
    """Get top process variants by frequency."""
    variants = defaultdict(int)
    
    for case_id, group in df.groupby("case_id"):
        trace = " → ".join(group.sort_values("timestamp")["activity"].tolist())
        variants[trace] += 1
    
    # Sort by frequency
    sorted_variants = sorted(variants.items(), key=lambda x: -x[1])
    
    result = []
    total_cases = sum(variants.values())
    
    for i, (trace, count) in enumerate(sorted_variants[:top_n], 1):
        result.append({
            "rank": i,
            "trace": trace,
            "count": count,
            "percentage": round(count / total_cases * 100, 1)
        })
    
    return result


def generate_dfg_graphviz(
    dfg: Dict[Tuple[str, str], int],
    activity_counts: Dict[str, int],
    durations: Dict[str, float],
    bottleneck_activities: List[str] = None
) -> str:
    """
    Generate Graphviz DOT string for DFG visualization.
    
    Args:
        dfg: Dict mapping (source, target) to frequency
        activity_counts: Dict mapping activity to count
        durations: Dict mapping activity to average duration
        bottleneck_activities: List of activity names that are bottlenecks
    
    Returns:
        DOT string for Graphviz
    """
    if bottleneck_activities is None:
        bottleneck_activities = []
    
    lines = [
        'digraph DFG {',
        '    rankdir=TB;',
        '    node [shape=box, style="rounded,filled", fontname="Arial", fontsize=10];',
        '    edge [fontname="Arial", fontsize=9];',
        ''
    ]
    
    # Add nodes
    for activity, count in activity_counts.items():
        duration = durations.get(activity, 0)
        
        # Format duration
        if duration >= 24:
            dur_str = f"{duration/24:.1f}d"
        else:
            dur_str = f"{duration:.1f}h"
        
        # Style based on bottleneck status
        if activity in bottleneck_activities:
            fill_color = "#FFCDD2"  # Light red
            border_color = "#C62828"  # Dark red
            penwidth = "2"
        else:
            fill_color = "#E3F2FD"  # Light blue
            border_color = "#1565C0"  # Dark blue
            penwidth = "1"
        
        label = f"{activity}\\n({count} events, {dur_str})"
        lines.append(f'    "{activity}" [label="{label}", fillcolor="{fill_color}", color="{border_color}", penwidth={penwidth}];')
    
    lines.append('')
    
    # Add edges
    max_freq = max(dfg.values()) if dfg else 1
    
    for (source, target), freq in dfg.items():
        # Edge thickness based on frequency
        penwidth = max(1, (freq / max_freq) * 4)
        lines.append(f'    "{source}" -> "{target}" [label="{freq}", penwidth={penwidth:.1f}];')
    
    lines.append('}')
    
    return '\n'.join(lines)


def calculate_process_metrics(df: pd.DataFrame) -> Dict:
    """Calculate overall process metrics."""
    # Case durations
    case_durations = []
    for case_id, group in df.groupby("case_id"):
        sorted_group = group.sort_values("timestamp")
        start = sorted_group["timestamp"].min()
        end = sorted_group["timestamp"].max()
        duration_hours = (end - start).total_seconds() / 3600
        case_durations.append(duration_hours)
    
    unique_activities = df["activity"].nunique()
    total_events = len(df)
    total_cases = df["case_id"].nunique()
    
    return {
        "total_cases": total_cases,
        "total_events": total_events,
        "unique_activities": unique_activities,
        "avg_case_duration_hours": round(np.mean(case_durations), 2) if case_durations else 0,
        "median_case_duration_hours": round(np.median(case_durations), 2) if case_durations else 0,
        "min_case_duration_hours": round(min(case_durations), 2) if case_durations else 0,
        "max_case_duration_hours": round(max(case_durations), 2) if case_durations else 0,
        "events_per_case": round(total_events / total_cases, 1) if total_cases else 0
    }


# ============================================
# Conformance Twin - Pydantic Models & Functions
# ============================================

class ConformanceStatus(str, Enum):
    """Conformance check status levels."""
    COMPLIANT = "compliant"
    MINOR_DEVIATION = "minor_deviation"
    MAJOR_DEVIATION = "major_deviation"
    NON_COMPLIANT = "non_compliant"


class ActivityDeviation(BaseModel):
    """Model for individual activity deviation."""
    activity: str = Field(..., description="Activity name")
    deviation_type: str = Field(..., description="Type of deviation: missing, unexpected, sequence_error, timing_breach")
    expected_value: Optional[str] = Field(None, description="Expected value from SOP")
    actual_value: Optional[str] = Field(None, description="Actual observed value")
    severity: str = Field(..., description="Severity: low, medium, high, critical")
    description: str = Field(..., description="Human-readable description of deviation")


class SOPActivity(BaseModel):
    """Model for SOP activity definition."""
    name: str = Field(..., description="Activity name")
    sequence_order: int = Field(..., description="Expected sequence order (1-based)")
    mandatory: bool = Field(default=True, description="Whether activity is mandatory")
    max_duration_hours: Optional[float] = Field(None, description="Maximum allowed duration in hours")
    allowed_predecessors: List[str] = Field(default_factory=list, description="Allowed predecessor activities")
    allowed_successors: List[str] = Field(default_factory=list, description="Allowed successor activities")


class SOPSchema(BaseModel):
    """Model for Standard Operating Procedure schema."""
    sop_id: str = Field(..., description="Unique SOP identifier")
    sop_name: str = Field(..., description="SOP name")
    version: str = Field(default="1.0", description="SOP version")
    process_type: str = Field(..., description="Process type: loan_approval, fraud_check, kyc, etc.")
    activities: List[SOPActivity] = Field(..., description="List of activities in order")
    max_total_duration_hours: Optional[float] = Field(None, description="Maximum total process duration")
    strict_sequence: bool = Field(default=False, description="Whether sequence must be strictly followed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ConformanceResult(BaseModel):
    """Model for conformance check result."""
    case_id: str = Field(..., description="Case identifier")
    sop_id: str = Field(..., description="SOP identifier compared against")
    status: ConformanceStatus = Field(..., description="Overall conformance status")
    conformance_score: float = Field(..., ge=0, le=100, description="Conformance score (0-100)")
    deviations: List[ActivityDeviation] = Field(default_factory=list, description="List of deviations found")
    missing_activities: List[str] = Field(default_factory=list, description="Mandatory activities not found")
    unexpected_activities: List[str] = Field(default_factory=list, description="Activities not in SOP")
    sequence_violations: List[str] = Field(default_factory=list, description="Sequence order violations")
    timing_breaches: List[str] = Field(default_factory=list, description="Duration threshold breaches")
    actual_duration_hours: float = Field(..., description="Actual process duration")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")


class ConformanceSummary(BaseModel):
    """Model for aggregated conformance analysis."""
    sop_id: str = Field(..., description="SOP identifier")
    total_cases: int = Field(..., description="Total cases analyzed")
    compliant_cases: int = Field(..., description="Number of fully compliant cases")
    minor_deviation_cases: int = Field(..., description="Cases with minor deviations")
    major_deviation_cases: int = Field(..., description="Cases with major deviations")
    non_compliant_cases: int = Field(..., description="Non-compliant cases")
    avg_conformance_score: float = Field(..., description="Average conformance score")
    most_common_deviations: List[str] = Field(default_factory=list, description="Most frequent deviation types")
    problem_activities: List[str] = Field(default_factory=list, description="Activities with most issues")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")


# Default Loan Approval SOP Schema
LOAN_APPROVAL_SOP = SOPSchema(
    sop_id="SOP-LOAN-001",
    sop_name="Loan Approval Standard Operating Procedure",
    version="2.0",
    process_type="loan_approval",
    activities=[
        SOPActivity(
            name="Application Received",
            sequence_order=1,
            mandatory=True,
            max_duration_hours=4,
            allowed_predecessors=[],
            allowed_successors=["Document Verification"]
        ),
        SOPActivity(
            name="Document Verification",
            sequence_order=2,
            mandatory=True,
            max_duration_hours=24,
            allowed_predecessors=["Application Received"],
            allowed_successors=["Credit Check"]
        ),
        SOPActivity(
            name="Credit Check",
            sequence_order=3,
            mandatory=True,
            max_duration_hours=8,
            allowed_predecessors=["Document Verification"],
            allowed_successors=["Risk Assessment"]
        ),
        SOPActivity(
            name="Risk Assessment",
            sequence_order=4,
            mandatory=True,
            max_duration_hours=48,
            allowed_predecessors=["Credit Check"],
            allowed_successors=["Manager Approval", "Escalation Review"]
        ),
        SOPActivity(
            name="Manager Approval",
            sequence_order=5,
            mandatory=True,
            max_duration_hours=12,
            allowed_predecessors=["Risk Assessment", "Escalation Review"],
            allowed_successors=["Final Review", "Escalation Review"]
        ),
        SOPActivity(
            name="Escalation Review",
            sequence_order=5,
            mandatory=False,
            max_duration_hours=24,
            allowed_predecessors=["Risk Assessment", "Manager Approval"],
            allowed_successors=["Manager Approval", "Final Review"]
        ),
        SOPActivity(
            name="Final Review",
            sequence_order=6,
            mandatory=True,
            max_duration_hours=6,
            allowed_predecessors=["Manager Approval", "Escalation Review"],
            allowed_successors=["Loan Disbursement"]
        ),
        SOPActivity(
            name="Loan Disbursement",
            sequence_order=7,
            mandatory=True,
            max_duration_hours=4,
            allowed_predecessors=["Final Review"],
            allowed_successors=[]
        ),
    ],
    max_total_duration_hours=120,  # 5 days
    strict_sequence=False,  # Allow escalation loops
    metadata={
        "regulation": "POJK 35/2018",
        "department": "Credit Operations",
        "last_updated": "2024-01-15"
    }
)


def load_sop_schema(json_path: Optional[str] = None) -> SOPSchema:
    """
    Load SOP schema from JSON file or return default Loan Approval SOP.

    Args:
        json_path: Path to JSON file containing SOP schema

    Returns:
        SOPSchema object
    """
    if json_path is None:
        return LOAN_APPROVAL_SOP

    with open(json_path, "r") as f:
        data = json.load(f)
    return SOPSchema(**data)


def check_case_conformance(
    case_events: pd.DataFrame,
    sop: SOPSchema
) -> ConformanceResult:
    """
    Check conformance of a single case against SOP schema.

    Args:
        case_events: DataFrame with columns [activity, timestamp] for a single case
        sop: SOP schema to compare against

    Returns:
        ConformanceResult object
    """
    case_id = case_events["case_id"].iloc[0] if "case_id" in case_events.columns else "unknown"

    # Sort by timestamp
    sorted_events = case_events.sort_values("timestamp")
    actual_activities = sorted_events["activity"].tolist()
    timestamps = sorted_events["timestamp"].tolist()

    # Calculate actual duration
    if len(timestamps) >= 2:
        actual_duration = (timestamps[-1] - timestamps[0]).total_seconds() / 3600
    else:
        actual_duration = 0

    deviations = []
    missing_activities = []
    unexpected_activities = []
    sequence_violations = []
    timing_breaches = []

    # Build SOP activity lookup
    sop_activities = {a.name: a for a in sop.activities}
    mandatory_activities = {a.name for a in sop.activities if a.mandatory}
    all_sop_activities = {a.name for a in sop.activities}

    # Check for missing mandatory activities
    actual_set = set(actual_activities)
    for mandatory in mandatory_activities:
        if mandatory not in actual_set:
            missing_activities.append(mandatory)
            deviations.append(ActivityDeviation(
                activity=mandatory,
                deviation_type="missing",
                expected_value=mandatory,
                actual_value=None,
                severity="high" if mandatory in ["Application Received", "Loan Disbursement"] else "medium",
                description=f"Mandatory activity '{mandatory}' not found in case"
            ))

    # Check for unexpected activities
    for activity in actual_activities:
        if activity not in all_sop_activities:
            unexpected_activities.append(activity)
            deviations.append(ActivityDeviation(
                activity=activity,
                deviation_type="unexpected",
                expected_value=None,
                actual_value=activity,
                severity="low",
                description=f"Activity '{activity}' not defined in SOP"
            ))

    # Check sequence violations
    prev_activity = None
    for i, activity in enumerate(actual_activities):
        if activity in sop_activities and prev_activity in sop_activities:
            sop_prev = sop_activities[prev_activity]
            if sop_prev.allowed_successors and activity not in sop_prev.allowed_successors:
                violation = f"{prev_activity} -> {activity}"
                sequence_violations.append(violation)
                deviations.append(ActivityDeviation(
                    activity=activity,
                    deviation_type="sequence_error",
                    expected_value=f"Expected one of: {', '.join(sop_prev.allowed_successors)}",
                    actual_value=activity,
                    severity="medium",
                    description=f"'{activity}' should not follow '{prev_activity}' per SOP"
                ))
        prev_activity = activity

    # Check timing breaches (activity durations)
    for i in range(len(actual_activities) - 1):
        activity = actual_activities[i]
        if activity in sop_activities:
            sop_activity = sop_activities[activity]
            if sop_activity.max_duration_hours:
                duration = (timestamps[i + 1] - timestamps[i]).total_seconds() / 3600
                if duration > sop_activity.max_duration_hours:
                    timing_breaches.append(activity)
                    deviations.append(ActivityDeviation(
                        activity=activity,
                        deviation_type="timing_breach",
                        expected_value=f"<= {sop_activity.max_duration_hours}h",
                        actual_value=f"{duration:.1f}h",
                        severity="high" if duration > sop_activity.max_duration_hours * 2 else "medium",
                        description=f"'{activity}' took {duration:.1f}h, exceeding limit of {sop_activity.max_duration_hours}h"
                    ))

    # Check total duration
    if sop.max_total_duration_hours and actual_duration > sop.max_total_duration_hours:
        deviations.append(ActivityDeviation(
            activity="TOTAL_PROCESS",
            deviation_type="timing_breach",
            expected_value=f"<= {sop.max_total_duration_hours}h",
            actual_value=f"{actual_duration:.1f}h",
            severity="high",
            description=f"Total process duration {actual_duration:.1f}h exceeds limit of {sop.max_total_duration_hours}h"
        ))

    # Calculate conformance score
    total_checks = len(mandatory_activities) + len(actual_activities) + 1  # +1 for total duration
    issues = len(missing_activities) + len(unexpected_activities) + len(sequence_violations) + len(timing_breaches)
    conformance_score = max(0, 100 - (issues / max(total_checks, 1)) * 100)

    # Determine status
    if conformance_score >= 95 and len(deviations) == 0:
        status = ConformanceStatus.COMPLIANT
    elif conformance_score >= 80 and not any(d.severity in ["high", "critical"] for d in deviations):
        status = ConformanceStatus.MINOR_DEVIATION
    elif conformance_score >= 60:
        status = ConformanceStatus.MAJOR_DEVIATION
    else:
        status = ConformanceStatus.NON_COMPLIANT

    return ConformanceResult(
        case_id=case_id,
        sop_id=sop.sop_id,
        status=status,
        conformance_score=round(conformance_score, 2),
        deviations=deviations,
        missing_activities=missing_activities,
        unexpected_activities=unexpected_activities,
        sequence_violations=sequence_violations,
        timing_breaches=timing_breaches,
        actual_duration_hours=round(actual_duration, 2)
    )


def conformance_twin_analysis(
    df: pd.DataFrame,
    sop: Optional[SOPSchema] = None,
    sop_json_path: Optional[str] = None
) -> Tuple[ConformanceSummary, List[ConformanceResult]]:
    """
    Perform Conformance Twin analysis comparing DFG/event log against SOP schema.

    Args:
        df: Event log DataFrame with columns [case_id, activity, timestamp]
        sop: SOPSchema object (uses default Loan Approval if None)
        sop_json_path: Path to JSON file with SOP schema (alternative to sop parameter)

    Returns:
        Tuple of (ConformanceSummary, List[ConformanceResult])
    """
    # Load SOP
    if sop is None:
        sop = load_sop_schema(sop_json_path)

    # Analyze each case
    results = []
    for case_id, case_events in df.groupby("case_id"):
        result = check_case_conformance(case_events, sop)
        results.append(result)

    # Calculate summary statistics
    compliant = sum(1 for r in results if r.status == ConformanceStatus.COMPLIANT)
    minor = sum(1 for r in results if r.status == ConformanceStatus.MINOR_DEVIATION)
    major = sum(1 for r in results if r.status == ConformanceStatus.MAJOR_DEVIATION)
    non_compliant = sum(1 for r in results if r.status == ConformanceStatus.NON_COMPLIANT)

    avg_score = sum(r.conformance_score for r in results) / len(results) if results else 0

    # Find most common deviations and problem activities
    deviation_counts: Dict[str, int] = defaultdict(int)
    activity_issue_counts: Dict[str, int] = defaultdict(int)

    for result in results:
        for dev in result.deviations:
            deviation_counts[dev.deviation_type] += 1
            activity_issue_counts[dev.activity] += 1

    most_common = sorted(deviation_counts.keys(), key=lambda x: -deviation_counts[x])[:5]
    problem_activities = sorted(activity_issue_counts.keys(), key=lambda x: -activity_issue_counts[x])[:5]

    summary = ConformanceSummary(
        sop_id=sop.sop_id,
        total_cases=len(results),
        compliant_cases=compliant,
        minor_deviation_cases=minor,
        major_deviation_cases=major,
        non_compliant_cases=non_compliant,
        avg_conformance_score=round(avg_score, 2),
        most_common_deviations=most_common,
        problem_activities=problem_activities
    )

    return summary, results


def export_sop_to_json(sop: SOPSchema, output_path: str) -> str:
    """
    Export SOP schema to JSON file.

    Args:
        sop: SOPSchema object
        output_path: Path to save JSON file

    Returns:
        Path to saved file
    """
    with open(output_path, "w") as f:
        json.dump(sop.model_dump(), f, indent=2, default=str)
    return output_path


# Export
__all__ = [
    "generate_sample_event_log",
    "parse_event_log",
    "calculate_dfg",
    "calculate_activity_durations",
    "detect_bottlenecks",
    "get_process_variants",
    "generate_dfg_graphviz",
    "calculate_process_metrics",
    "BottleneckInfo",
    # Conformance Twin exports
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
]
