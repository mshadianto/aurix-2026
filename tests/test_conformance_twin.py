"""
Tests for Conformance Twin functionality in process_mining module.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta

from modules.process_mining import (
    # Core functions
    generate_sample_event_log,
    calculate_dfg,
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


class TestSOPSchemaModels:
    """Test SOP schema Pydantic models."""

    def test_sop_activity_creation(self):
        """Test SOPActivity model creation."""
        activity = SOPActivity(
            name="Test Activity",
            sequence_order=1,
            mandatory=True,
            max_duration_hours=24,
            allowed_predecessors=["Start"],
            allowed_successors=["End"]
        )

        assert activity.name == "Test Activity"
        assert activity.sequence_order == 1
        assert activity.mandatory is True
        assert activity.max_duration_hours == 24
        assert "Start" in activity.allowed_predecessors
        assert "End" in activity.allowed_successors

    def test_sop_schema_creation(self):
        """Test SOPSchema model creation."""
        schema = SOPSchema(
            sop_id="TEST-001",
            sop_name="Test SOP",
            version="1.0",
            process_type="test_process",
            activities=[
                SOPActivity(name="Step 1", sequence_order=1),
                SOPActivity(name="Step 2", sequence_order=2),
            ],
            max_total_duration_hours=48,
            strict_sequence=True
        )

        assert schema.sop_id == "TEST-001"
        assert len(schema.activities) == 2
        assert schema.max_total_duration_hours == 48
        assert schema.strict_sequence is True

    def test_loan_approval_sop_exists(self):
        """Test that default Loan Approval SOP is properly defined."""
        assert LOAN_APPROVAL_SOP.sop_id == "SOP-LOAN-001"
        assert len(LOAN_APPROVAL_SOP.activities) >= 7
        assert LOAN_APPROVAL_SOP.process_type == "loan_approval"

        # Check mandatory activities exist
        activity_names = [a.name for a in LOAN_APPROVAL_SOP.activities]
        assert "Application Received" in activity_names
        assert "Loan Disbursement" in activity_names


class TestConformanceResultModels:
    """Test conformance result Pydantic models."""

    def test_activity_deviation_creation(self):
        """Test ActivityDeviation model creation."""
        deviation = ActivityDeviation(
            activity="Credit Check",
            deviation_type="missing",
            expected_value="Credit Check",
            actual_value=None,
            severity="high",
            description="Mandatory activity not found"
        )

        assert deviation.activity == "Credit Check"
        assert deviation.deviation_type == "missing"
        assert deviation.severity == "high"

    def test_conformance_result_creation(self):
        """Test ConformanceResult model creation."""
        result = ConformanceResult(
            case_id="LOAN-001",
            sop_id="SOP-LOAN-001",
            status=ConformanceStatus.COMPLIANT,
            conformance_score=95.0,
            deviations=[],
            missing_activities=[],
            unexpected_activities=[],
            sequence_violations=[],
            timing_breaches=[],
            actual_duration_hours=48.5
        )

        assert result.case_id == "LOAN-001"
        assert result.status == ConformanceStatus.COMPLIANT
        assert result.conformance_score == 95.0
        assert len(result.deviations) == 0

    def test_conformance_score_validation(self):
        """Test that conformance score must be between 0 and 100."""
        with pytest.raises(ValueError):
            ConformanceResult(
                case_id="TEST",
                sop_id="SOP",
                status=ConformanceStatus.COMPLIANT,
                conformance_score=150.0,  # Invalid - over 100
                actual_duration_hours=0
            )


class TestLoadSOPSchema:
    """Test SOP schema loading functionality."""

    def test_load_default_sop(self):
        """Test loading default Loan Approval SOP."""
        sop = load_sop_schema(None)

        assert sop.sop_id == LOAN_APPROVAL_SOP.sop_id
        assert sop.sop_name == LOAN_APPROVAL_SOP.sop_name

    def test_load_sop_from_dict(self, tmp_path):
        """Test loading SOP from JSON file."""
        import json

        sop_data = {
            "sop_id": "TEST-SOP",
            "sop_name": "Test SOP",
            "process_type": "test",
            "activities": [
                {"name": "Step A", "sequence_order": 1},
                {"name": "Step B", "sequence_order": 2},
            ]
        }

        json_path = tmp_path / "test_sop.json"
        with open(json_path, "w") as f:
            json.dump(sop_data, f)

        sop = load_sop_schema(str(json_path))

        assert sop.sop_id == "TEST-SOP"
        assert len(sop.activities) == 2


class TestCheckCaseConformance:
    """Test case conformance checking."""

    def test_compliant_case(self):
        """Test a case that follows SOP correctly."""
        base_time = datetime.now()
        events = pd.DataFrame([
            {"case_id": "LOAN-001", "activity": "Application Received", "timestamp": base_time},
            {"case_id": "LOAN-001", "activity": "Document Verification", "timestamp": base_time + timedelta(hours=2)},
            {"case_id": "LOAN-001", "activity": "Credit Check", "timestamp": base_time + timedelta(hours=6)},
            {"case_id": "LOAN-001", "activity": "Risk Assessment", "timestamp": base_time + timedelta(hours=12)},
            {"case_id": "LOAN-001", "activity": "Manager Approval", "timestamp": base_time + timedelta(hours=36)},
            {"case_id": "LOAN-001", "activity": "Final Review", "timestamp": base_time + timedelta(hours=40)},
            {"case_id": "LOAN-001", "activity": "Loan Disbursement", "timestamp": base_time + timedelta(hours=42)},
        ])
        events["timestamp"] = pd.to_datetime(events["timestamp"])

        result = check_case_conformance(events, LOAN_APPROVAL_SOP)

        assert result.case_id == "LOAN-001"
        assert result.conformance_score >= 80
        assert len(result.missing_activities) == 0

    def test_missing_mandatory_activity(self):
        """Test detection of missing mandatory activity."""
        base_time = datetime.now()
        events = pd.DataFrame([
            {"case_id": "LOAN-002", "activity": "Application Received", "timestamp": base_time},
            {"case_id": "LOAN-002", "activity": "Document Verification", "timestamp": base_time + timedelta(hours=2)},
            # Missing Credit Check
            {"case_id": "LOAN-002", "activity": "Risk Assessment", "timestamp": base_time + timedelta(hours=12)},
            {"case_id": "LOAN-002", "activity": "Manager Approval", "timestamp": base_time + timedelta(hours=36)},
            {"case_id": "LOAN-002", "activity": "Final Review", "timestamp": base_time + timedelta(hours=40)},
            {"case_id": "LOAN-002", "activity": "Loan Disbursement", "timestamp": base_time + timedelta(hours=42)},
        ])
        events["timestamp"] = pd.to_datetime(events["timestamp"])

        result = check_case_conformance(events, LOAN_APPROVAL_SOP)

        assert "Credit Check" in result.missing_activities
        assert len([d for d in result.deviations if d.deviation_type == "missing"]) > 0

    def test_unexpected_activity(self):
        """Test detection of unexpected activity not in SOP."""
        base_time = datetime.now()
        events = pd.DataFrame([
            {"case_id": "LOAN-003", "activity": "Application Received", "timestamp": base_time},
            {"case_id": "LOAN-003", "activity": "Custom Step", "timestamp": base_time + timedelta(hours=1)},  # Not in SOP
            {"case_id": "LOAN-003", "activity": "Document Verification", "timestamp": base_time + timedelta(hours=2)},
            {"case_id": "LOAN-003", "activity": "Credit Check", "timestamp": base_time + timedelta(hours=6)},
            {"case_id": "LOAN-003", "activity": "Risk Assessment", "timestamp": base_time + timedelta(hours=12)},
            {"case_id": "LOAN-003", "activity": "Manager Approval", "timestamp": base_time + timedelta(hours=36)},
            {"case_id": "LOAN-003", "activity": "Final Review", "timestamp": base_time + timedelta(hours=40)},
            {"case_id": "LOAN-003", "activity": "Loan Disbursement", "timestamp": base_time + timedelta(hours=42)},
        ])
        events["timestamp"] = pd.to_datetime(events["timestamp"])

        result = check_case_conformance(events, LOAN_APPROVAL_SOP)

        assert "Custom Step" in result.unexpected_activities

    def test_timing_breach(self):
        """Test detection of timing breaches."""
        base_time = datetime.now()
        events = pd.DataFrame([
            {"case_id": "LOAN-004", "activity": "Application Received", "timestamp": base_time},
            {"case_id": "LOAN-004", "activity": "Document Verification", "timestamp": base_time + timedelta(hours=48)},  # Exceeds 24h limit
            {"case_id": "LOAN-004", "activity": "Credit Check", "timestamp": base_time + timedelta(hours=52)},
            {"case_id": "LOAN-004", "activity": "Risk Assessment", "timestamp": base_time + timedelta(hours=60)},
            {"case_id": "LOAN-004", "activity": "Manager Approval", "timestamp": base_time + timedelta(hours=84)},
            {"case_id": "LOAN-004", "activity": "Final Review", "timestamp": base_time + timedelta(hours=88)},
            {"case_id": "LOAN-004", "activity": "Loan Disbursement", "timestamp": base_time + timedelta(hours=90)},
        ])
        events["timestamp"] = pd.to_datetime(events["timestamp"])

        result = check_case_conformance(events, LOAN_APPROVAL_SOP)

        # Application Received took 48 hours but max is 4 hours
        assert len(result.timing_breaches) > 0
        assert any(d.deviation_type == "timing_breach" for d in result.deviations)


class TestConformanceTwinAnalysis:
    """Test full conformance twin analysis."""

    def test_analysis_with_sample_data(self):
        """Test conformance analysis with sample event log."""
        event_log = generate_sample_event_log(num_cases=20)

        summary, results = conformance_twin_analysis(event_log)

        assert isinstance(summary, ConformanceSummary)
        assert len(results) == 20
        assert summary.total_cases == 20
        assert summary.avg_conformance_score >= 0
        assert summary.avg_conformance_score <= 100

    def test_analysis_status_counts(self):
        """Test that status counts sum to total cases."""
        event_log = generate_sample_event_log(num_cases=50)

        summary, results = conformance_twin_analysis(event_log)

        total_from_counts = (
            summary.compliant_cases +
            summary.minor_deviation_cases +
            summary.major_deviation_cases +
            summary.non_compliant_cases
        )
        assert total_from_counts == summary.total_cases


class TestExportSOPToJSON:
    """Test SOP export functionality."""

    def test_export_and_reload(self, tmp_path):
        """Test that exported SOP can be reloaded correctly."""
        output_path = tmp_path / "exported_sop.json"

        export_sop_to_json(LOAN_APPROVAL_SOP, str(output_path))

        # Reload and verify
        reloaded = load_sop_schema(str(output_path))

        assert reloaded.sop_id == LOAN_APPROVAL_SOP.sop_id
        assert reloaded.sop_name == LOAN_APPROVAL_SOP.sop_name
        assert len(reloaded.activities) == len(LOAN_APPROVAL_SOP.activities)
