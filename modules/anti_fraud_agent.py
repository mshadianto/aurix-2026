"""
Anti-Fraud Agent Module for AURIX 2026.
Pattern recognition for AML (Anti-Money Laundering) and APU-PPT (Anti Pencucian Uang -
Pencegahan Pendanaan Terorisme) compliance per Indonesian regulations.

Features:
- Transaction pattern analysis for suspicious activity detection
- Structuring/Smurfing detection (transactions split to avoid reporting thresholds)
- Rapid movement of funds detection
- Unusual activity patterns (dormant accounts, velocity anomalies)
- LLM-powered narrative analysis for SAR (Suspicious Activity Report)
- Pydantic schema enforcement for all outputs

Regulations:
- POJK 12/2017: Implementation of AML-CFT Program
- PBI 19/2017: AML-CFT Implementation for Banks
- UU 8/2010: Prevention and Eradication of Money Laundering
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from pydantic import BaseModel, Field, field_validator
import pandas as pd
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# ============================================
# Pydantic Models for AML/APU-PPT
# ============================================

class RiskLevel(str, Enum):
    """Risk level classification for suspicious activity."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of AML/APU-PPT alerts."""
    STRUCTURING = "structuring"  # Split transactions to avoid threshold
    RAPID_MOVEMENT = "rapid_movement"  # Quick in-out of funds
    VELOCITY_ANOMALY = "velocity_anomaly"  # Unusual transaction frequency
    DORMANT_REACTIVATION = "dormant_reactivation"  # Dormant account suddenly active
    LARGE_CASH = "large_cash"  # Large cash transactions
    ROUND_AMOUNT = "round_amount"  # Suspicious round amounts
    HIGH_RISK_JURISDICTION = "high_risk_jurisdiction"  # Transactions to/from high-risk countries
    PEP_RELATED = "pep_related"  # Politically Exposed Person involvement
    SHELL_COMPANY = "shell_company"  # Potential shell company patterns
    LAYERING = "layering"  # Complex transaction patterns to obscure origin
    UNUSUAL_BEHAVIOR = "unusual_behavior"  # Behavior inconsistent with customer profile


class TransactionRecord(BaseModel):
    """Model for individual transaction record."""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    account_id: str = Field(..., description="Account identifier")
    customer_id: str = Field(..., description="Customer identifier")
    transaction_date: datetime = Field(..., description="Transaction timestamp")
    transaction_type: str = Field(..., description="Type: deposit, withdrawal, transfer_in, transfer_out")
    amount: float = Field(..., ge=0, description="Transaction amount in IDR")
    currency: str = Field(default="IDR", description="Currency code")
    counterparty_account: Optional[str] = Field(None, description="Counterparty account if applicable")
    counterparty_name: Optional[str] = Field(None, description="Counterparty name")
    counterparty_country: Optional[str] = Field(None, description="Counterparty country code")
    channel: str = Field(default="branch", description="Channel: branch, atm, mobile, internet")
    branch_id: Optional[str] = Field(None, description="Branch identifier")
    description: Optional[str] = Field(None, description="Transaction description/narration")
    is_cash: bool = Field(default=False, description="Whether transaction involves cash")


class CustomerProfile(BaseModel):
    """Model for customer profile information."""
    customer_id: str = Field(..., description="Customer identifier")
    customer_name: str = Field(..., description="Customer full name")
    customer_type: str = Field(default="individual", description="Type: individual, corporate")
    risk_rating: str = Field(default="medium", description="KYC risk rating: low, medium, high")
    occupation: Optional[str] = Field(None, description="Customer occupation")
    industry: Optional[str] = Field(None, description="Industry sector for corporate")
    monthly_income: Optional[float] = Field(None, description="Declared monthly income")
    account_open_date: Optional[datetime] = Field(None, description="Account opening date")
    is_pep: bool = Field(default=False, description="Is Politically Exposed Person")
    pep_relation: Optional[str] = Field(None, description="PEP relation type if applicable")
    nationality: str = Field(default="ID", description="Nationality country code")
    residence_country: str = Field(default="ID", description="Country of residence")


class AlertEvidence(BaseModel):
    """Model for evidence supporting an alert."""
    evidence_type: str = Field(..., description="Type of evidence")
    description: str = Field(..., description="Description of the evidence")
    transaction_ids: List[str] = Field(default_factory=list, description="Related transaction IDs")
    value: Optional[str] = Field(None, description="Quantitative value if applicable")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score 0-1")


class FraudAlert(BaseModel):
    """Model for AML/APU-PPT fraud alert."""
    alert_id: str = Field(..., description="Unique alert identifier")
    customer_id: str = Field(..., description="Customer identifier")
    account_id: str = Field(..., description="Account identifier")
    alert_type: AlertType = Field(..., description="Type of alert")
    risk_level: RiskLevel = Field(..., description="Risk level")
    alert_score: float = Field(..., ge=0, le=100, description="Alert score 0-100")
    title: str = Field(..., description="Alert title")
    description: str = Field(..., description="Detailed description")
    evidence: List[AlertEvidence] = Field(default_factory=list, description="Supporting evidence")
    transaction_ids: List[str] = Field(default_factory=list, description="Related transaction IDs")
    total_amount: float = Field(..., description="Total amount involved")
    detection_date: datetime = Field(default_factory=datetime.now, description="When alert was generated")
    regulatory_reference: str = Field(..., description="Relevant regulation reference")
    recommended_action: str = Field(..., description="Recommended action")


class SARNarrative(BaseModel):
    """Model for Suspicious Activity Report narrative."""
    alert_id: str = Field(..., description="Related alert ID")
    narrative_type: str = Field(default="full", description="Type: summary, full, executive")
    subject_info: str = Field(..., description="Subject identification information")
    activity_summary: str = Field(..., description="Summary of suspicious activity")
    timeline: str = Field(..., description="Chronological timeline of events")
    red_flags: List[str] = Field(default_factory=list, description="Identified red flags")
    regulatory_violations: List[str] = Field(default_factory=list, description="Potential regulatory violations")
    recommendation: str = Field(..., description="Filing recommendation")
    generated_by: str = Field(default="AURIX AI Agent", description="Generator identification")
    generated_at: datetime = Field(default_factory=datetime.now, description="Generation timestamp")


class FraudAnalysisSummary(BaseModel):
    """Model for aggregated fraud analysis summary."""
    analysis_id: str = Field(..., description="Analysis identifier")
    analysis_period_start: datetime = Field(..., description="Analysis period start")
    analysis_period_end: datetime = Field(..., description="Analysis period end")
    total_transactions_analyzed: int = Field(..., description="Total transactions analyzed")
    total_customers_analyzed: int = Field(..., description="Total customers analyzed")
    total_alerts_generated: int = Field(..., description="Total alerts generated")
    alerts_by_type: Dict[str, int] = Field(default_factory=dict, description="Alert count by type")
    alerts_by_risk: Dict[str, int] = Field(default_factory=dict, description="Alert count by risk level")
    total_amount_flagged: float = Field(..., description="Total amount in flagged transactions")
    top_risk_customers: List[str] = Field(default_factory=list, description="Top 10 high-risk customers")
    detection_rate: float = Field(..., description="Detection rate percentage")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")


# ============================================
# Configuration Constants (Indonesian Regulations)
# ============================================

# PPATK reporting threshold (IDR 500 million cash or IDR 100 million transfer to high-risk)
CASH_THRESHOLD_IDR = 500_000_000  # IDR 500 million
TRANSFER_THRESHOLD_IDR = 100_000_000  # IDR 100 million for suspicious
STRUCTURING_THRESHOLD_IDR = 400_000_000  # 80% of cash threshold

# High-risk jurisdictions (FATF grey/black list + known high-risk)
HIGH_RISK_COUNTRIES = {
    "KP", "IR", "MM", "SY", "YE", "AF", "PK",  # FATF blacklist/greylist
    "PA", "VG", "KY", "BZ", "SC"  # Known tax havens
}

# Velocity thresholds
MAX_DAILY_TRANSACTIONS = 15
MAX_WEEKLY_TRANSACTIONS = 50
DORMANT_THRESHOLD_DAYS = 180


# ============================================
# Anti-Fraud Agent Class
# ============================================

class AntiFraudAgent:
    """
    Anti-Fraud Agent for AML/APU-PPT pattern recognition.
    Uses rule-based detection enhanced with LLM narrative generation.
    """

    def __init__(self, llm_client=None):
        """
        Initialize the Anti-Fraud Agent.

        Args:
            llm_client: Optional LLMClient instance for narrative generation
        """
        self.llm_client = llm_client
        self._alert_counter = 0

    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        self._alert_counter += 1
        return f"AML-{datetime.now().strftime('%Y%m%d')}-{self._alert_counter:05d}"

    def analyze_transactions(
        self,
        transactions: List[TransactionRecord],
        customer_profiles: Optional[Dict[str, CustomerProfile]] = None
    ) -> Tuple[FraudAnalysisSummary, List[FraudAlert]]:
        """
        Analyze transactions for AML/APU-PPT patterns.

        Args:
            transactions: List of transaction records
            customer_profiles: Optional dict of customer_id -> CustomerProfile

        Returns:
            Tuple of (FraudAnalysisSummary, List[FraudAlert])
        """
        if not transactions:
            return self._create_empty_summary(), []

        alerts: List[FraudAlert] = []

        # Group transactions by customer and account
        by_customer: Dict[str, List[TransactionRecord]] = defaultdict(list)
        by_account: Dict[str, List[TransactionRecord]] = defaultdict(list)

        for txn in transactions:
            by_customer[txn.customer_id].append(txn)
            by_account[txn.account_id].append(txn)

        # Run detection rules
        for customer_id, customer_txns in by_customer.items():
            profile = customer_profiles.get(customer_id) if customer_profiles else None

            # Structuring detection
            structuring_alerts = self._detect_structuring(customer_txns, profile)
            alerts.extend(structuring_alerts)

            # Rapid movement detection
            rapid_alerts = self._detect_rapid_movement(customer_txns, profile)
            alerts.extend(rapid_alerts)

            # Velocity anomaly detection
            velocity_alerts = self._detect_velocity_anomaly(customer_txns, profile)
            alerts.extend(velocity_alerts)

            # Large cash transaction detection
            large_cash_alerts = self._detect_large_cash(customer_txns, profile)
            alerts.extend(large_cash_alerts)

            # High-risk jurisdiction detection
            jurisdiction_alerts = self._detect_high_risk_jurisdiction(customer_txns, profile)
            alerts.extend(jurisdiction_alerts)

            # Round amount detection
            round_alerts = self._detect_round_amounts(customer_txns, profile)
            alerts.extend(round_alerts)

        # Dormant account detection (per account)
        for account_id, account_txns in by_account.items():
            dormant_alerts = self._detect_dormant_reactivation(account_txns)
            alerts.extend(dormant_alerts)

        # Build summary
        summary = self._build_summary(transactions, alerts)

        return summary, alerts

    def _detect_structuring(
        self,
        transactions: List[TransactionRecord],
        profile: Optional[CustomerProfile]
    ) -> List[FraudAlert]:
        """Detect structuring/smurfing patterns (split transactions to avoid threshold)."""
        alerts = []

        # Group transactions by date
        by_date: Dict[str, List[TransactionRecord]] = defaultdict(list)
        for txn in transactions:
            date_key = txn.transaction_date.strftime("%Y-%m-%d")
            by_date[date_key].append(txn)

        # Check each day for structuring
        for date_str, daily_txns in by_date.items():
            cash_txns = [t for t in daily_txns if t.is_cash and t.transaction_type in ["deposit", "withdrawal"]]

            if len(cash_txns) >= 2:
                total_cash = sum(t.amount for t in cash_txns)

                # Check if individual amounts are below threshold but total exceeds
                max_single = max(t.amount for t in cash_txns)
                if (total_cash >= STRUCTURING_THRESHOLD_IDR and
                        max_single < CASH_THRESHOLD_IDR and
                        len(cash_txns) >= 3):

                    alert = FraudAlert(
                        alert_id=self._generate_alert_id(),
                        customer_id=transactions[0].customer_id,
                        account_id=transactions[0].account_id,
                        alert_type=AlertType.STRUCTURING,
                        risk_level=RiskLevel.HIGH,
                        alert_score=85.0,
                        title=f"Potential Structuring Detected on {date_str}",
                        description=(
                            f"Customer made {len(cash_txns)} cash transactions totaling "
                            f"IDR {total_cash:,.0f} on {date_str}, with each transaction "
                            f"below the IDR {CASH_THRESHOLD_IDR:,.0f} reporting threshold. "
                            f"This pattern is consistent with structuring to avoid CTR filing."
                        ),
                        evidence=[
                            AlertEvidence(
                                evidence_type="transaction_pattern",
                                description=f"{len(cash_txns)} transactions below threshold",
                                transaction_ids=[t.transaction_id for t in cash_txns],
                                value=f"Total: IDR {total_cash:,.0f}",
                                confidence=0.85
                            )
                        ],
                        transaction_ids=[t.transaction_id for t in cash_txns],
                        total_amount=total_cash,
                        regulatory_reference="POJK 12/2017 Article 15 - Structuring Detection",
                        recommended_action="File STR within 3 business days per PPATK requirement"
                    )
                    alerts.append(alert)

        return alerts

    def _detect_rapid_movement(
        self,
        transactions: List[TransactionRecord],
        profile: Optional[CustomerProfile]
    ) -> List[FraudAlert]:
        """Detect rapid movement of funds (quick in-out pattern)."""
        alerts = []

        # Sort by date
        sorted_txns = sorted(transactions, key=lambda t: t.transaction_date)

        # Look for large deposits followed by quick withdrawals/transfers
        for i, txn in enumerate(sorted_txns):
            if txn.transaction_type in ["deposit", "transfer_in"] and txn.amount >= 100_000_000:
                # Look for outflows within 48 hours
                window_end = txn.transaction_date + timedelta(hours=48)
                subsequent = [
                    t for t in sorted_txns[i + 1:]
                    if t.transaction_date <= window_end
                    and t.transaction_type in ["withdrawal", "transfer_out"]
                ]

                if subsequent:
                    outflow_total = sum(t.amount for t in subsequent)
                    if outflow_total >= txn.amount * 0.8:  # 80% or more moved out

                        alert = FraudAlert(
                            alert_id=self._generate_alert_id(),
                            customer_id=txn.customer_id,
                            account_id=txn.account_id,
                            alert_type=AlertType.RAPID_MOVEMENT,
                            risk_level=RiskLevel.HIGH,
                            alert_score=80.0,
                            title="Rapid Movement of Funds Detected",
                            description=(
                                f"Large inflow of IDR {txn.amount:,.0f} on "
                                f"{txn.transaction_date.strftime('%Y-%m-%d')} followed by "
                                f"rapid outflow of IDR {outflow_total:,.0f} within 48 hours. "
                                f"This pattern may indicate layering or pass-through activity."
                            ),
                            evidence=[
                                AlertEvidence(
                                    evidence_type="timing_pattern",
                                    description="Funds moved within 48 hours",
                                    transaction_ids=[txn.transaction_id] + [t.transaction_id for t in subsequent],
                                    value=f"{(outflow_total / txn.amount * 100):.1f}% moved out",
                                    confidence=0.80
                                )
                            ],
                            transaction_ids=[txn.transaction_id] + [t.transaction_id for t in subsequent],
                            total_amount=txn.amount + outflow_total,
                            regulatory_reference="PBI 19/2017 - Unusual Transaction Patterns",
                            recommended_action="Enhanced due diligence and potential STR filing"
                        )
                        alerts.append(alert)

        return alerts

    def _detect_velocity_anomaly(
        self,
        transactions: List[TransactionRecord],
        profile: Optional[CustomerProfile]
    ) -> List[FraudAlert]:
        """Detect unusual transaction velocity."""
        alerts = []

        # Group by week
        by_week: Dict[str, List[TransactionRecord]] = defaultdict(list)
        for txn in transactions:
            week_key = txn.transaction_date.strftime("%Y-W%W")
            by_week[week_key].append(txn)

        for week_str, weekly_txns in by_week.items():
            if len(weekly_txns) > MAX_WEEKLY_TRANSACTIONS:
                alert = FraudAlert(
                    alert_id=self._generate_alert_id(),
                    customer_id=transactions[0].customer_id,
                    account_id=transactions[0].account_id,
                    alert_type=AlertType.VELOCITY_ANOMALY,
                    risk_level=RiskLevel.MEDIUM,
                    alert_score=65.0,
                    title=f"High Transaction Velocity in Week {week_str}",
                    description=(
                        f"Customer executed {len(weekly_txns)} transactions in one week, "
                        f"exceeding the typical threshold of {MAX_WEEKLY_TRANSACTIONS}. "
                        f"This unusual activity warrants review."
                    ),
                    evidence=[
                        AlertEvidence(
                            evidence_type="velocity_metric",
                            description=f"{len(weekly_txns)} transactions vs {MAX_WEEKLY_TRANSACTIONS} threshold",
                            transaction_ids=[t.transaction_id for t in weekly_txns],
                            value=f"{len(weekly_txns)} transactions",
                            confidence=0.65
                        )
                    ],
                    transaction_ids=[t.transaction_id for t in weekly_txns],
                    total_amount=sum(t.amount for t in weekly_txns),
                    regulatory_reference="POJK 12/2017 - Unusual Activity Monitoring",
                    recommended_action="Review transaction purpose and verify with customer"
                )
                alerts.append(alert)

        return alerts

    def _detect_large_cash(
        self,
        transactions: List[TransactionRecord],
        profile: Optional[CustomerProfile]
    ) -> List[FraudAlert]:
        """Detect large cash transactions requiring CTR."""
        alerts = []

        for txn in transactions:
            if txn.is_cash and txn.amount >= CASH_THRESHOLD_IDR:
                alert = FraudAlert(
                    alert_id=self._generate_alert_id(),
                    customer_id=txn.customer_id,
                    account_id=txn.account_id,
                    alert_type=AlertType.LARGE_CASH,
                    risk_level=RiskLevel.MEDIUM,
                    alert_score=70.0,
                    title=f"Large Cash Transaction - CTR Required",
                    description=(
                        f"Cash {txn.transaction_type} of IDR {txn.amount:,.0f} "
                        f"exceeds IDR {CASH_THRESHOLD_IDR:,.0f} threshold. "
                        f"Cash Transaction Report (CTR) filing required per PPATK regulation."
                    ),
                    evidence=[
                        AlertEvidence(
                            evidence_type="threshold_breach",
                            description="Amount exceeds cash reporting threshold",
                            transaction_ids=[txn.transaction_id],
                            value=f"IDR {txn.amount:,.0f}",
                            confidence=1.0
                        )
                    ],
                    transaction_ids=[txn.transaction_id],
                    total_amount=txn.amount,
                    regulatory_reference="PP 43/2015 - Cash Transaction Reporting",
                    recommended_action="File CTR to PPATK within 14 days"
                )
                alerts.append(alert)

        return alerts

    def _detect_high_risk_jurisdiction(
        self,
        transactions: List[TransactionRecord],
        profile: Optional[CustomerProfile]
    ) -> List[FraudAlert]:
        """Detect transactions involving high-risk jurisdictions."""
        alerts = []

        for txn in transactions:
            if txn.counterparty_country and txn.counterparty_country in HIGH_RISK_COUNTRIES:
                alert = FraudAlert(
                    alert_id=self._generate_alert_id(),
                    customer_id=txn.customer_id,
                    account_id=txn.account_id,
                    alert_type=AlertType.HIGH_RISK_JURISDICTION,
                    risk_level=RiskLevel.HIGH,
                    alert_score=90.0,
                    title=f"Transaction to/from High-Risk Jurisdiction ({txn.counterparty_country})",
                    description=(
                        f"Transaction of IDR {txn.amount:,.0f} involves counterparty "
                        f"in {txn.counterparty_country}, which is on the FATF high-risk "
                        f"jurisdiction list. Enhanced due diligence required."
                    ),
                    evidence=[
                        AlertEvidence(
                            evidence_type="jurisdiction_risk",
                            description=f"Counterparty in {txn.counterparty_country}",
                            transaction_ids=[txn.transaction_id],
                            value=txn.counterparty_country,
                            confidence=0.95
                        )
                    ],
                    transaction_ids=[txn.transaction_id],
                    total_amount=txn.amount,
                    regulatory_reference="POJK 12/2017 Article 23 - High-Risk Countries",
                    recommended_action="Perform enhanced due diligence, verify transaction purpose, consider STR"
                )
                alerts.append(alert)

        return alerts

    def _detect_round_amounts(
        self,
        transactions: List[TransactionRecord],
        profile: Optional[CustomerProfile]
    ) -> List[FraudAlert]:
        """Detect suspicious round amount patterns."""
        alerts = []

        # Look for patterns of round amounts
        round_txns = []
        for txn in transactions:
            # Check if amount is a round million
            if txn.amount >= 10_000_000 and txn.amount % 10_000_000 == 0:
                round_txns.append(txn)

        # Alert if there are many round transactions
        if len(round_txns) >= 5:
            total_round = sum(t.amount for t in round_txns)
            alert = FraudAlert(
                alert_id=self._generate_alert_id(),
                customer_id=transactions[0].customer_id,
                account_id=transactions[0].account_id,
                alert_type=AlertType.ROUND_AMOUNT,
                risk_level=RiskLevel.LOW,
                alert_score=45.0,
                title="Pattern of Round Amount Transactions",
                description=(
                    f"Customer has {len(round_txns)} transactions with perfectly round "
                    f"amounts totaling IDR {total_round:,.0f}. Round amounts can indicate "
                    f"informal money transfer or structuring attempts."
                ),
                evidence=[
                    AlertEvidence(
                        evidence_type="amount_pattern",
                        description="Multiple round-amount transactions",
                        transaction_ids=[t.transaction_id for t in round_txns],
                        value=f"{len(round_txns)} transactions",
                        confidence=0.45
                    )
                ],
                transaction_ids=[t.transaction_id for t in round_txns],
                total_amount=total_round,
                regulatory_reference="POJK 12/2017 - Unusual Transaction Characteristics",
                recommended_action="Review for legitimate business purpose"
            )
            alerts.append(alert)

        return alerts

    def _detect_dormant_reactivation(
        self,
        transactions: List[TransactionRecord]
    ) -> List[FraudAlert]:
        """Detect sudden reactivation of dormant accounts."""
        alerts = []

        if len(transactions) < 2:
            return alerts

        # Sort by date
        sorted_txns = sorted(transactions, key=lambda t: t.transaction_date)

        # Check for large gaps followed by activity
        for i in range(1, len(sorted_txns)):
            prev_txn = sorted_txns[i - 1]
            curr_txn = sorted_txns[i]

            gap_days = (curr_txn.transaction_date - prev_txn.transaction_date).days

            if gap_days >= DORMANT_THRESHOLD_DAYS:
                # Check if reactivation involves significant amount
                subsequent_txns = sorted_txns[i:i + 5]  # Next 5 transactions
                total_amount = sum(t.amount for t in subsequent_txns)

                if total_amount >= 50_000_000:  # IDR 50 million threshold
                    alert = FraudAlert(
                        alert_id=self._generate_alert_id(),
                        customer_id=curr_txn.customer_id,
                        account_id=curr_txn.account_id,
                        alert_type=AlertType.DORMANT_REACTIVATION,
                        risk_level=RiskLevel.MEDIUM,
                        alert_score=60.0,
                        title=f"Dormant Account Reactivation After {gap_days} Days",
                        description=(
                            f"Account was dormant for {gap_days} days and suddenly became active "
                            f"with IDR {total_amount:,.0f} in transactions. Dormant account "
                            f"reactivation can indicate account takeover or money mule activity."
                        ),
                        evidence=[
                            AlertEvidence(
                                evidence_type="dormancy_pattern",
                                description=f"{gap_days} days of inactivity",
                                transaction_ids=[t.transaction_id for t in subsequent_txns],
                                value=f"{gap_days} days dormant",
                                confidence=0.60
                            )
                        ],
                        transaction_ids=[t.transaction_id for t in subsequent_txns],
                        total_amount=total_amount,
                        regulatory_reference="POJK 12/2017 - Dormant Account Monitoring",
                        recommended_action="Verify customer identity and transaction purpose"
                    )
                    alerts.append(alert)
                    break  # Only one dormant alert per account

        return alerts

    def _build_summary(
        self,
        transactions: List[TransactionRecord],
        alerts: List[FraudAlert]
    ) -> FraudAnalysisSummary:
        """Build analysis summary."""
        dates = [t.transaction_date for t in transactions]

        alerts_by_type = defaultdict(int)
        alerts_by_risk = defaultdict(int)
        for alert in alerts:
            alerts_by_type[alert.alert_type.value] += 1
            alerts_by_risk[alert.risk_level.value] += 1

        # Find top risk customers
        customer_alert_count: Dict[str, int] = defaultdict(int)
        for alert in alerts:
            customer_alert_count[alert.customer_id] += 1
        top_customers = sorted(customer_alert_count.keys(), key=lambda x: -customer_alert_count[x])[:10]

        total_flagged = sum(a.total_amount for a in alerts)
        total_txn_amount = sum(t.amount for t in transactions)
        detection_rate = (len(alerts) / len(transactions) * 100) if transactions else 0

        return FraudAnalysisSummary(
            analysis_id=f"ANA-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            analysis_period_start=min(dates) if dates else datetime.now(),
            analysis_period_end=max(dates) if dates else datetime.now(),
            total_transactions_analyzed=len(transactions),
            total_customers_analyzed=len(set(t.customer_id for t in transactions)),
            total_alerts_generated=len(alerts),
            alerts_by_type=dict(alerts_by_type),
            alerts_by_risk=dict(alerts_by_risk),
            total_amount_flagged=total_flagged,
            top_risk_customers=top_customers,
            detection_rate=round(detection_rate, 2)
        )

    def _create_empty_summary(self) -> FraudAnalysisSummary:
        """Create empty summary for no transactions."""
        return FraudAnalysisSummary(
            analysis_id=f"ANA-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            analysis_period_start=datetime.now(),
            analysis_period_end=datetime.now(),
            total_transactions_analyzed=0,
            total_customers_analyzed=0,
            total_alerts_generated=0,
            alerts_by_type={},
            alerts_by_risk={},
            total_amount_flagged=0,
            top_risk_customers=[],
            detection_rate=0
        )

    def generate_sar_narrative(self, alert: FraudAlert) -> SARNarrative:
        """
        Generate SAR (Suspicious Activity Report) narrative using LLM.

        Args:
            alert: FraudAlert to generate narrative for

        Returns:
            SARNarrative object
        """
        if self.llm_client:
            return self._generate_llm_narrative(alert)
        else:
            return self._generate_template_narrative(alert)

    def _generate_llm_narrative(self, alert: FraudAlert) -> SARNarrative:
        """Generate narrative using LLM."""
        system_prompt = """You are an AML compliance expert for an Indonesian bank.
Generate a professional Suspicious Activity Report (SAR) narrative based on the alert details provided.
The narrative should follow PPATK (Indonesian Financial Intelligence Unit) reporting guidelines.
Be factual, specific, and avoid speculation beyond the evidence provided."""

        prompt = f"""Generate a SAR narrative for the following alert:

Alert Type: {alert.alert_type.value}
Risk Level: {alert.risk_level.value}
Title: {alert.title}
Description: {alert.description}
Total Amount: IDR {alert.total_amount:,.0f}
Evidence:
{chr(10).join([f"- {e.description} (Confidence: {e.confidence:.0%})" for e in alert.evidence])}
Regulatory Reference: {alert.regulatory_reference}

Please provide:
1. Subject identification summary
2. Activity summary
3. Timeline of events
4. List of red flags
5. Potential regulatory violations
6. Filing recommendation
"""

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3
            )

            # Parse LLM response into structured narrative
            content = response.content

            return SARNarrative(
                alert_id=alert.alert_id,
                narrative_type="full",
                subject_info=f"Customer ID: {alert.customer_id}, Account: {alert.account_id}",
                activity_summary=content[:500] if len(content) > 500 else content,
                timeline=f"Detection Date: {alert.detection_date.strftime('%Y-%m-%d %H:%M:%S')}",
                red_flags=[e.description for e in alert.evidence],
                regulatory_violations=[alert.regulatory_reference],
                recommendation=alert.recommended_action,
                generated_by="AURIX AI Agent (LLM)"
            )
        except Exception as e:
            logger.error(f"LLM narrative generation failed: {e}")
            return self._generate_template_narrative(alert)

    def _generate_template_narrative(self, alert: FraudAlert) -> SARNarrative:
        """Generate narrative using template (fallback)."""
        red_flags = [e.description for e in alert.evidence]

        activity_summary = (
            f"On {alert.detection_date.strftime('%Y-%m-%d')}, the AURIX monitoring system "
            f"detected suspicious activity classified as '{alert.alert_type.value}' with "
            f"a risk level of '{alert.risk_level.value}'. {alert.description} "
            f"The total amount involved is IDR {alert.total_amount:,.0f}."
        )

        timeline = (
            f"Alert generated: {alert.detection_date.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Transactions involved: {len(alert.transaction_ids)}\n"
            f"Alert score: {alert.alert_score:.1f}/100"
        )

        return SARNarrative(
            alert_id=alert.alert_id,
            narrative_type="full",
            subject_info=f"Customer ID: {alert.customer_id}, Account: {alert.account_id}",
            activity_summary=activity_summary,
            timeline=timeline,
            red_flags=red_flags,
            regulatory_violations=[alert.regulatory_reference],
            recommendation=alert.recommended_action,
            generated_by="AURIX AI Agent (Template)"
        )


def generate_sample_transactions(
    num_customers: int = 10,
    transactions_per_customer: int = 50,
    include_suspicious: bool = True
) -> List[TransactionRecord]:
    """
    Generate sample transaction data for testing.

    Args:
        num_customers: Number of customers to generate
        transactions_per_customer: Average transactions per customer
        include_suspicious: Whether to include suspicious patterns

    Returns:
        List of TransactionRecord objects
    """
    import random

    transactions = []
    base_date = datetime.now() - timedelta(days=90)

    for cust_num in range(1, num_customers + 1):
        customer_id = f"CUST-{cust_num:04d}"
        account_id = f"ACC-{cust_num:04d}-001"

        current_date = base_date + timedelta(days=random.uniform(0, 30))

        # Determine if this customer should have suspicious patterns
        is_suspicious = include_suspicious and random.random() < 0.2  # 20% suspicious

        num_txns = random.randint(
            transactions_per_customer // 2,
            transactions_per_customer * 2
        )

        for txn_num in range(num_txns):
            txn_type = random.choice(["deposit", "withdrawal", "transfer_in", "transfer_out"])
            is_cash = txn_type in ["deposit", "withdrawal"] and random.random() < 0.3

            # Generate amount
            if is_suspicious and is_cash:
                # Structuring pattern - just below threshold
                amount = random.uniform(300_000_000, 450_000_000)
            elif is_suspicious and random.random() < 0.3:
                # Large amounts for rapid movement
                amount = random.uniform(100_000_000, 500_000_000)
            else:
                # Normal amounts
                amount = random.uniform(1_000_000, 50_000_000)

            # Round amounts for some suspicious transactions
            if is_suspicious and random.random() < 0.5:
                amount = round(amount / 10_000_000) * 10_000_000

            # High-risk country for some suspicious transfers
            counterparty_country = None
            if txn_type in ["transfer_in", "transfer_out"]:
                if is_suspicious and random.random() < 0.3:
                    counterparty_country = random.choice(list(HIGH_RISK_COUNTRIES))
                else:
                    counterparty_country = random.choice(["ID", "SG", "MY", "US", "AU"])

            transactions.append(TransactionRecord(
                transaction_id=f"TXN-{customer_id}-{txn_num:04d}",
                account_id=account_id,
                customer_id=customer_id,
                transaction_date=current_date,
                transaction_type=txn_type,
                amount=amount,
                is_cash=is_cash,
                counterparty_country=counterparty_country,
                channel=random.choice(["branch", "atm", "mobile", "internet"]),
                description=f"Sample transaction {txn_num}"
            ))

            # Advance time (rapid for suspicious, normal for others)
            if is_suspicious and random.random() < 0.5:
                current_date += timedelta(hours=random.uniform(1, 24))
            else:
                current_date += timedelta(hours=random.uniform(24, 168))

    return sorted(transactions, key=lambda t: t.transaction_date)


# Export
__all__ = [
    # Enums
    "RiskLevel",
    "AlertType",
    # Models
    "TransactionRecord",
    "CustomerProfile",
    "AlertEvidence",
    "FraudAlert",
    "SARNarrative",
    "FraudAnalysisSummary",
    # Constants
    "CASH_THRESHOLD_IDR",
    "TRANSFER_THRESHOLD_IDR",
    "HIGH_RISK_COUNTRIES",
    # Classes
    "AntiFraudAgent",
    # Functions
    "generate_sample_transactions",
]
