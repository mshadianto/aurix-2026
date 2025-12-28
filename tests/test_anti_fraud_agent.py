"""
Tests for Anti-Fraud Agent module (AML/APU-PPT).
"""

import pytest
from datetime import datetime, timedelta

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


class TestPydanticModels:
    """Test Pydantic model validation."""

    def test_transaction_record_creation(self):
        """Test TransactionRecord model creation."""
        txn = TransactionRecord(
            transaction_id="TXN-001",
            account_id="ACC-001",
            customer_id="CUST-001",
            transaction_date=datetime.now(),
            transaction_type="deposit",
            amount=1_000_000,
            is_cash=True,
            channel="branch"
        )

        assert txn.transaction_id == "TXN-001"
        assert txn.amount == 1_000_000
        assert txn.is_cash is True
        assert txn.currency == "IDR"  # Default

    def test_transaction_record_negative_amount(self):
        """Test that negative amounts are rejected."""
        with pytest.raises(ValueError):
            TransactionRecord(
                transaction_id="TXN-001",
                account_id="ACC-001",
                customer_id="CUST-001",
                transaction_date=datetime.now(),
                transaction_type="deposit",
                amount=-1_000_000  # Invalid negative amount
            )

    def test_customer_profile_creation(self):
        """Test CustomerProfile model creation."""
        profile = CustomerProfile(
            customer_id="CUST-001",
            customer_name="John Doe",
            customer_type="individual",
            risk_rating="medium",
            is_pep=False
        )

        assert profile.customer_id == "CUST-001"
        assert profile.is_pep is False
        assert profile.nationality == "ID"  # Default

    def test_fraud_alert_creation(self):
        """Test FraudAlert model creation."""
        alert = FraudAlert(
            alert_id="AML-001",
            customer_id="CUST-001",
            account_id="ACC-001",
            alert_type=AlertType.STRUCTURING,
            risk_level=RiskLevel.HIGH,
            alert_score=85.0,
            title="Test Alert",
            description="Test description",
            total_amount=500_000_000,
            regulatory_reference="POJK 12/2017",
            recommended_action="File STR"
        )

        assert alert.alert_type == AlertType.STRUCTURING
        assert alert.risk_level == RiskLevel.HIGH
        assert alert.alert_score == 85.0

    def test_fraud_alert_score_validation(self):
        """Test that alert score must be 0-100."""
        with pytest.raises(ValueError):
            FraudAlert(
                alert_id="AML-001",
                customer_id="CUST-001",
                account_id="ACC-001",
                alert_type=AlertType.STRUCTURING,
                risk_level=RiskLevel.HIGH,
                alert_score=150.0,  # Invalid - over 100
                title="Test",
                description="Test",
                total_amount=100,
                regulatory_reference="Test",
                recommended_action="Test"
            )


class TestConstants:
    """Test regulatory constants."""

    def test_cash_threshold(self):
        """Test cash threshold is IDR 500 million."""
        assert CASH_THRESHOLD_IDR == 500_000_000

    def test_transfer_threshold(self):
        """Test transfer threshold is IDR 100 million."""
        assert TRANSFER_THRESHOLD_IDR == 100_000_000

    def test_high_risk_countries(self):
        """Test high-risk countries include known jurisdictions."""
        assert "KP" in HIGH_RISK_COUNTRIES  # North Korea
        assert "IR" in HIGH_RISK_COUNTRIES  # Iran
        assert "ID" not in HIGH_RISK_COUNTRIES  # Indonesia should not be in list


class TestAntiFraudAgent:
    """Test AntiFraudAgent class."""

    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        return AntiFraudAgent(llm_client=None)

    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.llm_client is None
        assert agent._alert_counter == 0

    def test_generate_alert_id(self, agent):
        """Test alert ID generation."""
        id1 = agent._generate_alert_id()
        id2 = agent._generate_alert_id()

        assert id1.startswith("AML-")
        assert id1 != id2
        assert "00001" in id1
        assert "00002" in id2

    def test_analyze_empty_transactions(self, agent):
        """Test analysis with empty transaction list."""
        summary, alerts = agent.analyze_transactions([])

        assert summary.total_transactions_analyzed == 0
        assert summary.total_alerts_generated == 0
        assert len(alerts) == 0

    def test_detect_structuring(self, agent):
        """Test structuring detection."""
        base_time = datetime.now()
        customer_id = "CUST-001"
        account_id = "ACC-001"

        # Create structuring pattern - multiple cash transactions below threshold
        transactions = [
            TransactionRecord(
                transaction_id=f"TXN-{i}",
                account_id=account_id,
                customer_id=customer_id,
                transaction_date=base_time + timedelta(hours=i),
                transaction_type="deposit",
                amount=150_000_000,  # 150 million each, below 500m threshold
                is_cash=True
            )
            for i in range(4)  # 4 x 150m = 600m total
        ]

        summary, alerts = agent.analyze_transactions(transactions)

        structuring_alerts = [a for a in alerts if a.alert_type == AlertType.STRUCTURING]
        assert len(structuring_alerts) >= 1
        assert structuring_alerts[0].risk_level == RiskLevel.HIGH

    def test_detect_large_cash(self, agent):
        """Test large cash transaction detection."""
        txn = TransactionRecord(
            transaction_id="TXN-001",
            account_id="ACC-001",
            customer_id="CUST-001",
            transaction_date=datetime.now(),
            transaction_type="deposit",
            amount=600_000_000,  # 600 million - above threshold
            is_cash=True
        )

        summary, alerts = agent.analyze_transactions([txn])

        large_cash_alerts = [a for a in alerts if a.alert_type == AlertType.LARGE_CASH]
        assert len(large_cash_alerts) == 1
        assert "CTR" in large_cash_alerts[0].title

    def test_detect_high_risk_jurisdiction(self, agent):
        """Test high-risk jurisdiction detection."""
        txn = TransactionRecord(
            transaction_id="TXN-001",
            account_id="ACC-001",
            customer_id="CUST-001",
            transaction_date=datetime.now(),
            transaction_type="transfer_out",
            amount=50_000_000,
            counterparty_country="KP"  # North Korea
        )

        summary, alerts = agent.analyze_transactions([txn])

        jurisdiction_alerts = [a for a in alerts if a.alert_type == AlertType.HIGH_RISK_JURISDICTION]
        assert len(jurisdiction_alerts) == 1
        assert jurisdiction_alerts[0].risk_level == RiskLevel.HIGH

    def test_detect_rapid_movement(self, agent):
        """Test rapid movement of funds detection."""
        base_time = datetime.now()
        customer_id = "CUST-001"
        account_id = "ACC-001"

        transactions = [
            TransactionRecord(
                transaction_id="TXN-001",
                account_id=account_id,
                customer_id=customer_id,
                transaction_date=base_time,
                transaction_type="deposit",
                amount=200_000_000  # 200 million inflow
            ),
            TransactionRecord(
                transaction_id="TXN-002",
                account_id=account_id,
                customer_id=customer_id,
                transaction_date=base_time + timedelta(hours=12),  # Within 48 hours
                transaction_type="transfer_out",
                amount=180_000_000  # 90% moved out quickly
            )
        ]

        summary, alerts = agent.analyze_transactions(transactions)

        rapid_alerts = [a for a in alerts if a.alert_type == AlertType.RAPID_MOVEMENT]
        assert len(rapid_alerts) >= 1

    def test_detect_velocity_anomaly(self, agent):
        """Test velocity anomaly detection."""
        # Use a fixed date to ensure all transactions are in the same week
        base_time = datetime(2024, 6, 10, 9, 0, 0)  # Monday
        customer_id = "CUST-001"
        account_id = "ACC-001"

        # Create many transactions in one week (all within 7 days)
        transactions = [
            TransactionRecord(
                transaction_id=f"TXN-{i}",
                account_id=account_id,
                customer_id=customer_id,
                transaction_date=base_time + timedelta(hours=i * 2),  # 2 hours apart, all in same week
                transaction_type="deposit",
                amount=1_000_000
            )
            for i in range(60)  # 60 transactions over ~5 days - exceeds weekly threshold of 50
        ]

        summary, alerts = agent.analyze_transactions(transactions)

        velocity_alerts = [a for a in alerts if a.alert_type == AlertType.VELOCITY_ANOMALY]
        assert len(velocity_alerts) >= 1

    def test_detect_dormant_reactivation(self, agent):
        """Test dormant account reactivation detection."""
        base_time = datetime.now()
        customer_id = "CUST-001"
        account_id = "ACC-001"

        transactions = [
            TransactionRecord(
                transaction_id="TXN-001",
                account_id=account_id,
                customer_id=customer_id,
                transaction_date=base_time - timedelta(days=200),  # 200 days ago
                transaction_type="deposit",
                amount=1_000_000
            ),
            TransactionRecord(
                transaction_id="TXN-002",
                account_id=account_id,
                customer_id=customer_id,
                transaction_date=base_time,  # Now - after 200 day gap
                transaction_type="deposit",
                amount=100_000_000  # Large reactivation
            )
        ]

        summary, alerts = agent.analyze_transactions(transactions)

        dormant_alerts = [a for a in alerts if a.alert_type == AlertType.DORMANT_REACTIVATION]
        assert len(dormant_alerts) >= 1


class TestSARNarrativeGeneration:
    """Test SAR narrative generation."""

    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        return AntiFraudAgent(llm_client=None)

    @pytest.fixture
    def sample_alert(self):
        """Create sample alert."""
        return FraudAlert(
            alert_id="AML-TEST-001",
            customer_id="CUST-001",
            account_id="ACC-001",
            alert_type=AlertType.STRUCTURING,
            risk_level=RiskLevel.HIGH,
            alert_score=85.0,
            title="Test Structuring Alert",
            description="Multiple cash deposits below threshold",
            evidence=[
                AlertEvidence(
                    evidence_type="transaction_pattern",
                    description="4 transactions below threshold",
                    transaction_ids=["TXN-1", "TXN-2", "TXN-3", "TXN-4"],
                    value="Total: IDR 600,000,000",
                    confidence=0.85
                )
            ],
            transaction_ids=["TXN-1", "TXN-2", "TXN-3", "TXN-4"],
            total_amount=600_000_000,
            regulatory_reference="POJK 12/2017 Article 15",
            recommended_action="File STR within 3 business days"
        )

    def test_generate_template_narrative(self, agent, sample_alert):
        """Test template-based narrative generation (no LLM)."""
        narrative = agent.generate_sar_narrative(sample_alert)

        assert isinstance(narrative, SARNarrative)
        assert narrative.alert_id == sample_alert.alert_id
        assert "CUST-001" in narrative.subject_info
        assert len(narrative.red_flags) > 0
        assert "Template" in narrative.generated_by


class TestSampleDataGeneration:
    """Test sample transaction generation."""

    def test_generate_sample_transactions(self):
        """Test sample transaction generation."""
        transactions = generate_sample_transactions(
            num_customers=5,
            transactions_per_customer=20,
            include_suspicious=True
        )

        assert len(transactions) > 0
        assert all(isinstance(t, TransactionRecord) for t in transactions)

        # Verify different customers
        customer_ids = set(t.customer_id for t in transactions)
        assert len(customer_ids) == 5

    def test_sample_transactions_sorted(self):
        """Test that sample transactions are sorted by date."""
        transactions = generate_sample_transactions(num_customers=3)

        for i in range(len(transactions) - 1):
            assert transactions[i].transaction_date <= transactions[i + 1].transaction_date


class TestFraudAnalysisSummary:
    """Test fraud analysis summary."""

    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        return AntiFraudAgent(llm_client=None)

    def test_summary_statistics(self, agent):
        """Test summary statistics are calculated correctly."""
        transactions = generate_sample_transactions(
            num_customers=10,
            transactions_per_customer=30,
            include_suspicious=True
        )

        summary, alerts = agent.analyze_transactions(transactions)

        assert isinstance(summary, FraudAnalysisSummary)
        assert summary.total_transactions_analyzed == len(transactions)
        assert summary.total_customers_analyzed == 10
        assert summary.total_alerts_generated == len(alerts)

        # Verify alerts_by_type sums to total
        type_count = sum(summary.alerts_by_type.values())
        assert type_count == summary.total_alerts_generated

        # Verify alerts_by_risk sums to total
        risk_count = sum(summary.alerts_by_risk.values())
        assert risk_count == summary.total_alerts_generated

    def test_detection_rate(self, agent):
        """Test detection rate calculation."""
        transactions = generate_sample_transactions(
            num_customers=5,
            transactions_per_customer=10,
            include_suspicious=True
        )

        summary, alerts = agent.analyze_transactions(transactions)

        expected_rate = (len(alerts) / len(transactions) * 100) if transactions else 0
        assert abs(summary.detection_rate - round(expected_rate, 2)) < 0.1
