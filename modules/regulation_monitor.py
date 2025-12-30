"""
Regulation Monitor Module for AURIX 2026.
Tracks changes in OJK/BI regulations and generates compliance action items.

Features:
- Track OJK/BI regulation updates
- Diff comparison between regulation versions
- Impact assessment for policy changes
- Auto-generate compliance action items
- Notification of regulatory changes
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from pydantic import BaseModel, Field
import logging
import hashlib
import difflib

logger = logging.getLogger(__name__)


# ============================================
# Enums and Constants
# ============================================

class RegulatorType(str, Enum):
    """Indonesian financial regulators."""
    OJK = "ojk"  # Otoritas Jasa Keuangan
    BI = "bi"    # Bank Indonesia
    PPATK = "ppatk"  # Pusat Pelaporan dan Analisis Transaksi Keuangan
    LPS = "lps"  # Lembaga Penjamin Simpanan
    KSEI = "ksei"  # Kustodian Sentral Efek Indonesia


class RegulationType(str, Enum):
    """Types of regulations."""
    POJK = "pojk"  # Peraturan OJK
    SEOJK = "seojk"  # Surat Edaran OJK
    PBI = "pbi"  # Peraturan Bank Indonesia
    SEBI = "sebi"  # Surat Edaran BI
    PADG = "padg"  # Peraturan Anggota Dewan Gubernur
    UU = "uu"  # Undang-Undang
    PP = "pp"  # Peraturan Pemerintah


class ChangeType(str, Enum):
    """Type of regulatory change."""
    NEW = "new"  # New regulation
    AMENDMENT = "amendment"  # Amendment to existing
    REVOCATION = "revocation"  # Revocation
    CLARIFICATION = "clarification"  # Clarification/SE


class ImpactLevel(str, Enum):
    """Impact level of regulatory change."""
    LOW = "low"  # Minor administrative changes
    MEDIUM = "medium"  # Process/reporting changes
    HIGH = "high"  # Significant policy changes
    CRITICAL = "critical"  # Major restructuring required


class ComplianceStatus(str, Enum):
    """Compliance action status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    OVERDUE = "overdue"


# ============================================
# Pydantic Models
# ============================================

class RegulationSection(BaseModel):
    """Section within a regulation."""
    section_id: str = Field(..., description="Section identifier")
    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Section content")
    parent_section: Optional[str] = Field(None, description="Parent section ID")
    effective_date: Optional[datetime] = Field(None, description="Section effective date")


class RegulationDocument(BaseModel):
    """Complete regulation document."""
    regulation_id: str = Field(..., description="Regulation identifier (e.g., POJK 12/2017)")
    regulator: RegulatorType = Field(..., description="Issuing regulator")
    regulation_type: RegulationType = Field(..., description="Type of regulation")
    title: str = Field(..., description="Full regulation title")
    short_title: Optional[str] = Field(None, description="Short/common name")

    # Dates
    issue_date: datetime = Field(..., description="Date regulation was issued")
    effective_date: datetime = Field(..., description="Date regulation becomes effective")
    publication_date: Optional[datetime] = Field(None, description="Date published")

    # Content
    preamble: Optional[str] = Field(None, description="Regulation preamble")
    sections: List[RegulationSection] = Field(default_factory=list)
    full_text: str = Field(..., description="Full regulation text")

    # Metadata
    supersedes: List[str] = Field(default_factory=list, description="Regulations this supersedes")
    superseded_by: Optional[str] = Field(None, description="Regulation that supersedes this")
    related_regulations: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)

    # Versioning
    version: str = Field(default="1.0", description="Document version")
    content_hash: str = Field(..., description="Hash of content for change detection")

    last_updated: datetime = Field(default_factory=datetime.now)


class RegulationChange(BaseModel):
    """Detected change in regulation."""
    change_id: str = Field(..., description="Change identifier")
    regulation_id: str = Field(..., description="Affected regulation ID")
    regulation_title: str = Field(..., description="Regulation title")

    # Change details
    change_type: ChangeType = Field(..., description="Type of change")
    change_summary: str = Field(..., description="Summary of changes")
    affected_sections: List[str] = Field(default_factory=list)

    # Diff information
    old_version: Optional[str] = Field(None, description="Previous version")
    new_version: str = Field(..., description="New version")
    diff_details: List[Dict[str, str]] = Field(default_factory=list)

    # Impact
    impact_level: ImpactLevel = Field(..., description="Impact level")
    impact_areas: List[str] = Field(default_factory=list)
    affected_processes: List[str] = Field(default_factory=list)

    # Dates
    detected_date: datetime = Field(default_factory=datetime.now)
    effective_date: datetime = Field(..., description="When change becomes effective")
    compliance_deadline: Optional[datetime] = Field(None, description="Deadline for compliance")


class ComplianceActionItem(BaseModel):
    """Action item for compliance with regulatory change."""
    action_id: str = Field(..., description="Action identifier")
    change_id: str = Field(..., description="Related regulatory change")
    regulation_id: str = Field(..., description="Regulation ID")

    # Action details
    title: str = Field(..., description="Action title")
    description: str = Field(..., description="Detailed description")
    category: str = Field(..., description="Category: policy, process, system, training, reporting")

    # Assignment
    responsible_unit: str = Field(..., description="Responsible unit/department")
    owner: Optional[str] = Field(None, description="Action owner")

    # Timeline
    priority: str = Field(..., description="Priority: urgent, high, medium, low")
    due_date: datetime = Field(..., description="Due date")
    estimated_effort: str = Field(..., description="Estimated effort")

    # Status
    status: ComplianceStatus = Field(default=ComplianceStatus.PENDING)
    progress_percent: int = Field(default=0, ge=0, le=100)
    notes: List[str] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class RegulationChangeReport(BaseModel):
    """Report of regulatory changes over a period."""
    report_id: str = Field(..., description="Report identifier")
    report_period_start: datetime = Field(...)
    report_period_end: datetime = Field(...)

    # Summary
    total_changes: int = Field(default=0)
    new_regulations: int = Field(default=0)
    amendments: int = Field(default=0)
    revocations: int = Field(default=0)

    # By impact
    critical_changes: int = Field(default=0)
    high_impact_changes: int = Field(default=0)

    # Details
    changes: List[RegulationChange] = Field(default_factory=list)
    action_items: List[ComplianceActionItem] = Field(default_factory=list)

    # Analysis
    key_themes: List[str] = Field(default_factory=list)
    executive_summary: str = Field(..., description="Executive summary")

    generated_at: datetime = Field(default_factory=datetime.now)


# ============================================
# Regulation Monitor Engine
# ============================================

class RegulationMonitor:
    """
    Monitors regulatory changes and generates compliance action items.
    """

    def __init__(self):
        """Initialize the monitor."""
        self._regulations: Dict[str, RegulationDocument] = {}
        self._changes: List[RegulationChange] = []
        self._action_items: List[ComplianceActionItem] = []
        self._change_counter = 0
        self._action_counter = 0

    def _generate_change_id(self) -> str:
        """Generate unique change ID."""
        self._change_counter += 1
        return f"CHG-{datetime.now().strftime('%Y%m%d')}-{self._change_counter:05d}"

    def _generate_action_id(self) -> str:
        """Generate unique action ID."""
        self._action_counter += 1
        return f"ACT-{datetime.now().strftime('%Y%m%d')}-{self._action_counter:05d}"

    def _calculate_content_hash(self, content: str) -> str:
        """Calculate hash of regulation content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def register_regulation(self, regulation: RegulationDocument) -> None:
        """
        Register a regulation for monitoring.

        Args:
            regulation: Regulation document to register
        """
        if not regulation.content_hash:
            regulation.content_hash = self._calculate_content_hash(regulation.full_text)

        self._regulations[regulation.regulation_id] = regulation
        logger.info(f"Registered regulation: {regulation.regulation_id}")

    def check_for_changes(
        self,
        regulation_id: str,
        new_content: str,
        new_version: str
    ) -> Optional[RegulationChange]:
        """
        Check if regulation content has changed.

        Args:
            regulation_id: Regulation to check
            new_content: New content to compare
            new_version: New version number

        Returns:
            RegulationChange if change detected, None otherwise
        """
        if regulation_id not in self._regulations:
            return None

        old_reg = self._regulations[regulation_id]
        old_hash = old_reg.content_hash
        new_hash = self._calculate_content_hash(new_content)

        if old_hash == new_hash:
            return None

        # Detect changes using diff
        old_lines = old_reg.full_text.split('\n')
        new_lines = new_content.split('\n')

        differ = difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"{regulation_id} v{old_reg.version}",
            tofile=f"{regulation_id} v{new_version}",
            lineterm=""
        )

        diff_result = list(differ)

        # Analyze diff
        added_lines = [l for l in diff_result if l.startswith('+') and not l.startswith('+++')]
        removed_lines = [l for l in diff_result if l.startswith('-') and not l.startswith('---')]

        # Determine change type and impact
        if len(added_lines) > len(old_lines) * 0.5:
            change_type = ChangeType.AMENDMENT
            impact_level = ImpactLevel.HIGH
        elif len(removed_lines) > len(old_lines) * 0.3:
            change_type = ChangeType.AMENDMENT
            impact_level = ImpactLevel.MEDIUM
        else:
            change_type = ChangeType.AMENDMENT
            impact_level = ImpactLevel.LOW

        # Create change record
        change = RegulationChange(
            change_id=self._generate_change_id(),
            regulation_id=regulation_id,
            regulation_title=old_reg.title,
            change_type=change_type,
            change_summary=f"Detected {len(added_lines)} additions and {len(removed_lines)} removals",
            old_version=old_reg.version,
            new_version=new_version,
            diff_details=[
                {"type": "added", "count": str(len(added_lines))},
                {"type": "removed", "count": str(len(removed_lines))}
            ],
            impact_level=impact_level,
            effective_date=datetime.now() + timedelta(days=30)
        )

        self._changes.append(change)
        return change

    def detect_new_regulation(
        self,
        regulation: RegulationDocument
    ) -> RegulationChange:
        """
        Process a new regulation.

        Args:
            regulation: New regulation document

        Returns:
            RegulationChange for the new regulation
        """
        # Determine impact based on regulation type
        if regulation.regulation_type in [RegulationType.POJK, RegulationType.PBI, RegulationType.UU]:
            impact_level = ImpactLevel.HIGH
        elif regulation.regulation_type in [RegulationType.SEOJK, RegulationType.SEBI]:
            impact_level = ImpactLevel.MEDIUM
        else:
            impact_level = ImpactLevel.LOW

        change = RegulationChange(
            change_id=self._generate_change_id(),
            regulation_id=regulation.regulation_id,
            regulation_title=regulation.title,
            change_type=ChangeType.NEW,
            change_summary=f"New {regulation.regulation_type.value.upper()} issued by {regulation.regulator.value.upper()}",
            old_version=None,
            new_version=regulation.version,
            impact_level=impact_level,
            impact_areas=regulation.keywords,
            effective_date=regulation.effective_date,
            compliance_deadline=regulation.effective_date + timedelta(days=90)
        )

        self._changes.append(change)
        self.register_regulation(regulation)

        return change

    def generate_action_items(
        self,
        change: RegulationChange
    ) -> List[ComplianceActionItem]:
        """
        Generate compliance action items for a regulatory change.

        Args:
            change: Regulatory change

        Returns:
            List of action items
        """
        action_items = []

        # Base actions for any change
        if change.change_type == ChangeType.NEW:
            # New regulation requires comprehensive review
            action_items.append(ComplianceActionItem(
                action_id=self._generate_action_id(),
                change_id=change.change_id,
                regulation_id=change.regulation_id,
                title=f"Review {change.regulation_id}",
                description=f"Comprehensive review of new regulation: {change.regulation_title}",
                category="policy",
                responsible_unit="Compliance",
                priority="high" if change.impact_level in [ImpactLevel.HIGH, ImpactLevel.CRITICAL] else "medium",
                due_date=change.effective_date - timedelta(days=30),
                estimated_effort="2-3 days"
            ))

            action_items.append(ComplianceActionItem(
                action_id=self._generate_action_id(),
                change_id=change.change_id,
                regulation_id=change.regulation_id,
                title=f"Gap analysis for {change.regulation_id}",
                description="Identify gaps between current practices and new requirements",
                category="process",
                responsible_unit="Compliance",
                priority="high",
                due_date=change.effective_date - timedelta(days=21),
                estimated_effort="3-5 days"
            ))

            action_items.append(ComplianceActionItem(
                action_id=self._generate_action_id(),
                change_id=change.change_id,
                regulation_id=change.regulation_id,
                title=f"Update policies for {change.regulation_id}",
                description="Update internal policies to align with new regulation",
                category="policy",
                responsible_unit="Policy Team",
                priority="high",
                due_date=change.effective_date - timedelta(days=14),
                estimated_effort="5-7 days"
            ))

            action_items.append(ComplianceActionItem(
                action_id=self._generate_action_id(),
                change_id=change.change_id,
                regulation_id=change.regulation_id,
                title=f"Staff training on {change.regulation_id}",
                description="Train relevant staff on new regulatory requirements",
                category="training",
                responsible_unit="HR / Training",
                priority="medium",
                due_date=change.effective_date,
                estimated_effort="1-2 days"
            ))

        elif change.change_type == ChangeType.AMENDMENT:
            # Amendment requires focused review
            action_items.append(ComplianceActionItem(
                action_id=self._generate_action_id(),
                change_id=change.change_id,
                regulation_id=change.regulation_id,
                title=f"Review amendment to {change.regulation_id}",
                description=f"Review changes in version {change.new_version}",
                category="policy",
                responsible_unit="Compliance",
                priority="high" if change.impact_level == ImpactLevel.HIGH else "medium",
                due_date=change.effective_date - timedelta(days=14),
                estimated_effort="1-2 days"
            ))

            if change.impact_level in [ImpactLevel.HIGH, ImpactLevel.CRITICAL]:
                action_items.append(ComplianceActionItem(
                    action_id=self._generate_action_id(),
                    change_id=change.change_id,
                    regulation_id=change.regulation_id,
                    title=f"Impact assessment for {change.regulation_id} changes",
                    description="Assess impact of regulatory changes on current operations",
                    category="process",
                    responsible_unit="Operations",
                    priority="high",
                    due_date=change.effective_date - timedelta(days=7),
                    estimated_effort="2-3 days"
                ))

        elif change.change_type == ChangeType.REVOCATION:
            action_items.append(ComplianceActionItem(
                action_id=self._generate_action_id(),
                change_id=change.change_id,
                regulation_id=change.regulation_id,
                title=f"Archive {change.regulation_id}",
                description="Archive revoked regulation and update compliance matrix",
                category="policy",
                responsible_unit="Compliance",
                priority="low",
                due_date=change.effective_date + timedelta(days=30),
                estimated_effort="0.5 days"
            ))

        self._action_items.extend(action_items)
        return action_items

    def generate_change_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> RegulationChangeReport:
        """
        Generate report of regulatory changes.

        Args:
            start_date: Report period start
            end_date: Report period end

        Returns:
            RegulationChangeReport
        """
        # Filter changes in period
        period_changes = [
            c for c in self._changes
            if start_date <= c.detected_date <= end_date
        ]

        # Count by type
        new_count = sum(1 for c in period_changes if c.change_type == ChangeType.NEW)
        amendment_count = sum(1 for c in period_changes if c.change_type == ChangeType.AMENDMENT)
        revocation_count = sum(1 for c in period_changes if c.change_type == ChangeType.REVOCATION)

        # Count by impact
        critical_count = sum(1 for c in period_changes if c.impact_level == ImpactLevel.CRITICAL)
        high_count = sum(1 for c in period_changes if c.impact_level == ImpactLevel.HIGH)

        # Get related action items
        change_ids = {c.change_id for c in period_changes}
        related_actions = [a for a in self._action_items if a.change_id in change_ids]

        # Identify key themes
        all_keywords = []
        for c in period_changes:
            all_keywords.extend(c.impact_areas)
        theme_counts = {}
        for kw in all_keywords:
            theme_counts[kw] = theme_counts.get(kw, 0) + 1
        key_themes = sorted(theme_counts.keys(), key=lambda x: -theme_counts[x])[:5]

        # Generate summary
        summary = f"Regulatory changes from {start_date.strftime('%B %Y')} to {end_date.strftime('%B %Y')}: "
        summary += f"{len(period_changes)} total changes detected. "
        if new_count:
            summary += f"{new_count} new regulations issued. "
        if critical_count + high_count > 0:
            summary += f"{critical_count + high_count} high-impact changes require immediate attention. "
        summary += f"{len(related_actions)} action items generated."

        return RegulationChangeReport(
            report_id=f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            report_period_start=start_date,
            report_period_end=end_date,
            total_changes=len(period_changes),
            new_regulations=new_count,
            amendments=amendment_count,
            revocations=revocation_count,
            critical_changes=critical_count,
            high_impact_changes=high_count,
            changes=period_changes,
            action_items=related_actions,
            key_themes=key_themes,
            executive_summary=summary
        )

    def get_pending_actions(
        self,
        due_within_days: int = 30
    ) -> List[ComplianceActionItem]:
        """
        Get pending action items due within specified days.

        Args:
            due_within_days: Days to look ahead

        Returns:
            List of pending action items
        """
        cutoff = datetime.now() + timedelta(days=due_within_days)
        pending = [
            a for a in self._action_items
            if a.status in [ComplianceStatus.PENDING, ComplianceStatus.IN_PROGRESS]
            and a.due_date <= cutoff
        ]

        # Mark overdue items
        now = datetime.now()
        for action in pending:
            if action.due_date < now and action.status == ComplianceStatus.PENDING:
                action.status = ComplianceStatus.OVERDUE

        return sorted(pending, key=lambda x: x.due_date)


def generate_sample_regulations() -> List[RegulationDocument]:
    """Generate sample Indonesian financial regulations."""
    regulations = [
        RegulationDocument(
            regulation_id="POJK 12/2017",
            regulator=RegulatorType.OJK,
            regulation_type=RegulationType.POJK,
            title="Penerapan Program Anti Pencucian Uang dan Pencegahan Pendanaan Terorisme di Sektor Jasa Keuangan",
            short_title="APU-PPT",
            issue_date=datetime(2017, 11, 10),
            effective_date=datetime(2018, 1, 1),
            full_text="Peraturan tentang program APU-PPT...",
            keywords=["aml", "cft", "kyc", "cdd", "reporting"],
            content_hash="abc123"
        ),
        RegulationDocument(
            regulation_id="POJK 39/POJK.03/2019",
            regulator=RegulatorType.OJK,
            regulation_type=RegulationType.POJK,
            title="Penerapan Strategi Anti Fraud bagi Bank Umum",
            short_title="Anti-Fraud Strategy",
            issue_date=datetime(2019, 12, 20),
            effective_date=datetime(2020, 6, 20),
            full_text="Peraturan tentang strategi anti fraud...",
            keywords=["fraud", "risk", "prevention", "detection"],
            content_hash="def456"
        ),
        RegulationDocument(
            regulation_id="PBI 23/6/PBI/2021",
            regulator=RegulatorType.BI,
            regulation_type=RegulationType.PBI,
            title="Penyedia Jasa Pembayaran",
            short_title="PJP",
            issue_date=datetime(2021, 7, 1),
            effective_date=datetime(2021, 7, 1),
            full_text="Peraturan tentang penyedia jasa pembayaran...",
            keywords=["payment", "fintech", "license", "consumer protection"],
            content_hash="ghi789"
        ),
    ]
    return regulations


# Export
__all__ = [
    # Enums
    "RegulatorType",
    "RegulationType",
    "ChangeType",
    "ImpactLevel",
    "ComplianceStatus",
    # Models
    "RegulationSection",
    "RegulationDocument",
    "RegulationChange",
    "ComplianceActionItem",
    "RegulationChangeReport",
    # Classes
    "RegulationMonitor",
    # Functions
    "generate_sample_regulations",
]
