"""
Compliance Gap Analysis Module for AURIX 2026.
Maps controls to regulations and identifies coverage gaps.

Features:
- Control-to-regulation mapping
- Coverage gap identification
- Remediation roadmap generation
- Compliance score calculation
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


# ============================================
# Enums and Constants
# ============================================

class ControlStatus(str, Enum):
    """Control implementation status."""
    IMPLEMENTED = "implemented"
    PARTIALLY_IMPLEMENTED = "partially_implemented"
    NOT_IMPLEMENTED = "not_implemented"
    NOT_APPLICABLE = "not_applicable"


class ControlEffectiveness(str, Enum):
    """Control effectiveness rating."""
    EFFECTIVE = "effective"
    PARTIALLY_EFFECTIVE = "partially_effective"
    NOT_EFFECTIVE = "not_effective"
    NOT_TESTED = "not_tested"


class GapSeverity(str, Enum):
    """Gap severity level."""
    CRITICAL = "critical"  # Regulatory violation risk
    HIGH = "high"  # Significant deficiency
    MEDIUM = "medium"  # Improvement needed
    LOW = "low"  # Minor enhancement


class RemediationPriority(str, Enum):
    """Remediation priority."""
    URGENT = "urgent"  # Immediate action (< 30 days)
    HIGH = "high"  # Short-term (30-90 days)
    MEDIUM = "medium"  # Medium-term (90-180 days)
    LOW = "low"  # Long-term (> 180 days)


# ============================================
# Pydantic Models
# ============================================

class RegulatoryRequirement(BaseModel):
    """Single regulatory requirement."""
    requirement_id: str = Field(..., description="Requirement identifier")
    regulation_id: str = Field(..., description="Source regulation ID")
    regulation_name: str = Field(..., description="Regulation name")
    article: str = Field(..., description="Article/section reference")
    requirement_text: str = Field(..., description="Requirement text")
    category: str = Field(..., description="Category: governance, risk, control, reporting")
    mandatory: bool = Field(default=True, description="Is mandatory")
    effective_date: datetime = Field(..., description="When requirement became effective")


class Control(BaseModel):
    """Internal control definition."""
    control_id: str = Field(..., description="Control identifier")
    control_name: str = Field(..., description="Control name")
    description: str = Field(..., description="Control description")
    control_type: str = Field(..., description="Type: preventive, detective, corrective")
    control_category: str = Field(..., description="Category: manual, automated, hybrid")

    # Owner
    owner_department: str = Field(..., description="Owning department")
    owner_name: Optional[str] = Field(None, description="Control owner name")

    # Status
    status: ControlStatus = Field(default=ControlStatus.IMPLEMENTED)
    effectiveness: ControlEffectiveness = Field(default=ControlEffectiveness.NOT_TESTED)

    # Testing
    last_tested: Optional[datetime] = Field(None, description="Last test date")
    test_frequency: str = Field(default="annual", description="Test frequency")
    test_results: Optional[str] = Field(None, description="Latest test results")

    # Documentation
    procedure_reference: Optional[str] = Field(None, description="SOP reference")
    evidence_types: List[str] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class ControlMapping(BaseModel):
    """Mapping between control and regulatory requirement."""
    mapping_id: str = Field(..., description="Mapping identifier")
    control_id: str = Field(..., description="Control ID")
    requirement_id: str = Field(..., description="Requirement ID")

    # Coverage assessment
    coverage_type: str = Field(..., description="full, partial, indirect")
    coverage_percentage: float = Field(..., ge=0, le=100, description="Coverage percentage")
    coverage_notes: Optional[str] = Field(None, description="Coverage notes")

    # Validation
    validated: bool = Field(default=False, description="Mapping validated")
    validated_by: Optional[str] = Field(None, description="Validator")
    validated_date: Optional[datetime] = Field(None, description="Validation date")


class ComplianceGap(BaseModel):
    """Identified compliance gap."""
    gap_id: str = Field(..., description="Gap identifier")
    requirement_id: str = Field(..., description="Related requirement")
    regulation_id: str = Field(..., description="Related regulation")

    # Gap details
    gap_title: str = Field(..., description="Gap title")
    gap_description: str = Field(..., description="Detailed description")
    gap_type: str = Field(..., description="Type: no_control, partial_control, ineffective_control")
    severity: GapSeverity = Field(..., description="Gap severity")

    # Coverage
    current_coverage: float = Field(..., ge=0, le=100, description="Current coverage %")
    target_coverage: float = Field(default=100, ge=0, le=100, description="Target coverage %")
    coverage_gap: float = Field(..., ge=0, le=100, description="Coverage gap %")

    # Related controls
    related_controls: List[str] = Field(default_factory=list)

    # Impact
    potential_impact: str = Field(..., description="Potential impact if not addressed")
    risk_exposure: str = Field(..., description="Current risk exposure")

    # Remediation
    remediation_required: bool = Field(default=True)
    remediation_priority: RemediationPriority = Field(default=RemediationPriority.MEDIUM)

    identified_date: datetime = Field(default_factory=datetime.now)


class RemediationPlan(BaseModel):
    """Plan to remediate a compliance gap."""
    plan_id: str = Field(..., description="Plan identifier")
    gap_id: str = Field(..., description="Related gap ID")
    requirement_id: str = Field(..., description="Related requirement")

    # Plan details
    plan_title: str = Field(..., description="Plan title")
    plan_description: str = Field(..., description="Detailed plan")
    remediation_approach: str = Field(..., description="new_control, enhance_control, process_change")

    # Actions
    action_items: List[Dict[str, str]] = Field(default_factory=list)

    # Resources
    estimated_cost: Optional[str] = Field(None, description="Estimated cost")
    required_resources: List[str] = Field(default_factory=list)
    responsible_party: str = Field(..., description="Responsible party")

    # Timeline
    priority: RemediationPriority = Field(...)
    start_date: datetime = Field(...)
    target_completion: datetime = Field(...)
    milestones: List[Dict[str, Any]] = Field(default_factory=list)

    # Status
    status: str = Field(default="planned", description="planned, in_progress, completed, delayed")
    progress_percent: int = Field(default=0, ge=0, le=100)

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class ComplianceScorecard(BaseModel):
    """Overall compliance scorecard."""
    scorecard_id: str = Field(..., description="Scorecard identifier")
    assessment_date: datetime = Field(default_factory=datetime.now)

    # Overall scores
    overall_compliance_score: float = Field(..., ge=0, le=100, description="Overall score")
    overall_coverage: float = Field(..., ge=0, le=100, description="Overall coverage %")

    # By category
    category_scores: Dict[str, float] = Field(default_factory=dict)

    # By regulation
    regulation_scores: Dict[str, Dict[str, float]] = Field(default_factory=dict)

    # Gap summary
    total_requirements: int = Field(default=0)
    fully_covered: int = Field(default=0)
    partially_covered: int = Field(default=0)
    not_covered: int = Field(default=0)

    total_gaps: int = Field(default=0)
    critical_gaps: int = Field(default=0)
    high_gaps: int = Field(default=0)

    # Remediation status
    remediation_plans_total: int = Field(default=0)
    remediation_in_progress: int = Field(default=0)
    remediation_completed: int = Field(default=0)

    # Trend
    previous_score: Optional[float] = Field(None, description="Previous assessment score")
    score_trend: str = Field(default="stable", description="improving, stable, declining")

    # Narrative
    executive_summary: str = Field(..., description="Executive summary")
    key_findings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


# ============================================
# Compliance Gap Analyzer
# ============================================

class ComplianceGapAnalyzer:
    """
    Analyzes compliance gaps between controls and regulatory requirements.
    """

    def __init__(self):
        """Initialize the analyzer."""
        self._requirements: Dict[str, RegulatoryRequirement] = {}
        self._controls: Dict[str, Control] = {}
        self._mappings: List[ControlMapping] = []
        self._gaps: List[ComplianceGap] = []
        self._remediation_plans: List[RemediationPlan] = []
        self._gap_counter = 0
        self._plan_counter = 0

    def _generate_gap_id(self) -> str:
        """Generate unique gap ID."""
        self._gap_counter += 1
        return f"GAP-{datetime.now().strftime('%Y%m%d')}-{self._gap_counter:05d}"

    def _generate_plan_id(self) -> str:
        """Generate unique plan ID."""
        self._plan_counter += 1
        return f"REM-{datetime.now().strftime('%Y%m%d')}-{self._plan_counter:05d}"

    def register_requirement(self, requirement: RegulatoryRequirement) -> None:
        """Register a regulatory requirement."""
        self._requirements[requirement.requirement_id] = requirement

    def register_control(self, control: Control) -> None:
        """Register an internal control."""
        self._controls[control.control_id] = control

    def add_mapping(self, mapping: ControlMapping) -> None:
        """Add control-to-requirement mapping."""
        self._mappings.append(mapping)

    def analyze_coverage(
        self,
        requirement_id: str
    ) -> Tuple[float, List[str], Optional[ComplianceGap]]:
        """
        Analyze coverage for a specific requirement.

        Args:
            requirement_id: Requirement to analyze

        Returns:
            Tuple of (coverage_percentage, control_ids, gap if any)
        """
        if requirement_id not in self._requirements:
            return 0.0, [], None

        requirement = self._requirements[requirement_id]

        # Find mappings for this requirement
        mappings = [m for m in self._mappings if m.requirement_id == requirement_id]

        if not mappings:
            # No controls mapped - critical gap
            gap = ComplianceGap(
                gap_id=self._generate_gap_id(),
                requirement_id=requirement_id,
                regulation_id=requirement.regulation_id,
                gap_title=f"No control for: {requirement.article}",
                gap_description=f"No internal control mapped to requirement: {requirement.requirement_text[:100]}...",
                gap_type="no_control",
                severity=GapSeverity.CRITICAL if requirement.mandatory else GapSeverity.HIGH,
                current_coverage=0,
                target_coverage=100,
                coverage_gap=100,
                potential_impact="Regulatory non-compliance risk",
                risk_exposure="High - no mitigating control in place"
            )
            self._gaps.append(gap)
            return 0.0, [], gap

        # Calculate coverage
        total_coverage = 0
        control_ids = []

        for mapping in mappings:
            control = self._controls.get(mapping.control_id)
            if control:
                control_ids.append(mapping.control_id)

                # Adjust coverage based on control status and effectiveness
                base_coverage = mapping.coverage_percentage

                if control.status == ControlStatus.NOT_IMPLEMENTED:
                    effective_coverage = 0
                elif control.status == ControlStatus.PARTIALLY_IMPLEMENTED:
                    effective_coverage = base_coverage * 0.5
                else:
                    # Apply effectiveness factor
                    if control.effectiveness == ControlEffectiveness.EFFECTIVE:
                        effective_coverage = base_coverage
                    elif control.effectiveness == ControlEffectiveness.PARTIALLY_EFFECTIVE:
                        effective_coverage = base_coverage * 0.7
                    elif control.effectiveness == ControlEffectiveness.NOT_EFFECTIVE:
                        effective_coverage = base_coverage * 0.3
                    else:
                        effective_coverage = base_coverage * 0.5  # Not tested

                total_coverage += effective_coverage

        # Cap at 100%
        total_coverage = min(total_coverage, 100)

        # Determine if gap exists
        gap = None
        if total_coverage < 100:
            coverage_gap = 100 - total_coverage

            if coverage_gap >= 50:
                severity = GapSeverity.HIGH
                gap_type = "partial_control"
            elif coverage_gap >= 25:
                severity = GapSeverity.MEDIUM
                gap_type = "partial_control"
            else:
                severity = GapSeverity.LOW
                gap_type = "minor_gap"

            gap = ComplianceGap(
                gap_id=self._generate_gap_id(),
                requirement_id=requirement_id,
                regulation_id=requirement.regulation_id,
                gap_title=f"Partial coverage for: {requirement.article}",
                gap_description=f"Controls provide only {total_coverage:.0f}% coverage",
                gap_type=gap_type,
                severity=severity,
                current_coverage=total_coverage,
                target_coverage=100,
                coverage_gap=coverage_gap,
                related_controls=control_ids,
                potential_impact="Potential regulatory finding",
                risk_exposure=f"{severity.value.capitalize()} - {coverage_gap:.0f}% gap"
            )
            self._gaps.append(gap)

        return total_coverage, control_ids, gap

    def analyze_all_requirements(self) -> List[ComplianceGap]:
        """
        Analyze coverage for all registered requirements.

        Returns:
            List of identified gaps
        """
        self._gaps = []  # Reset gaps

        for req_id in self._requirements:
            self.analyze_coverage(req_id)

        return self._gaps

    def generate_remediation_plan(
        self,
        gap: ComplianceGap
    ) -> RemediationPlan:
        """
        Generate remediation plan for a gap.

        Args:
            gap: Compliance gap

        Returns:
            RemediationPlan
        """
        # Determine approach based on gap type
        if gap.gap_type == "no_control":
            approach = "new_control"
            action_items = [
                {"action": "Design new control", "owner": "Compliance"},
                {"action": "Develop control procedure", "owner": "Operations"},
                {"action": "Implement control", "owner": "Operations"},
                {"action": "Test control effectiveness", "owner": "Internal Audit"},
                {"action": "Document and train", "owner": "Training"}
            ]
            estimated_cost = "Medium-High"
        elif gap.gap_type == "partial_control":
            approach = "enhance_control"
            action_items = [
                {"action": "Review existing controls", "owner": "Compliance"},
                {"action": "Identify enhancement areas", "owner": "Operations"},
                {"action": "Implement enhancements", "owner": "Operations"},
                {"action": "Re-test effectiveness", "owner": "Internal Audit"}
            ]
            estimated_cost = "Medium"
        else:
            approach = "process_change"
            action_items = [
                {"action": "Review current process", "owner": "Operations"},
                {"action": "Implement process improvements", "owner": "Operations"},
                {"action": "Update documentation", "owner": "Operations"}
            ]
            estimated_cost = "Low"

        # Determine timeline based on priority
        now = datetime.now()
        if gap.remediation_priority == RemediationPriority.URGENT:
            target_days = 30
        elif gap.remediation_priority == RemediationPriority.HIGH:
            target_days = 90
        elif gap.remediation_priority == RemediationPriority.MEDIUM:
            target_days = 180
        else:
            target_days = 365

        plan = RemediationPlan(
            plan_id=self._generate_plan_id(),
            gap_id=gap.gap_id,
            requirement_id=gap.requirement_id,
            plan_title=f"Remediation: {gap.gap_title}",
            plan_description=f"Plan to address compliance gap: {gap.gap_description}",
            remediation_approach=approach,
            action_items=action_items,
            estimated_cost=estimated_cost,
            required_resources=["Compliance team", "Business unit", "IT support"],
            responsible_party="Compliance Department",
            priority=gap.remediation_priority,
            start_date=now,
            target_completion=now + timedelta(days=target_days),
            milestones=[
                {"milestone": "Planning complete", "target_date": (now + timedelta(days=target_days * 0.2)).isoformat()},
                {"milestone": "Implementation 50%", "target_date": (now + timedelta(days=target_days * 0.5)).isoformat()},
                {"milestone": "Testing complete", "target_date": (now + timedelta(days=target_days * 0.8)).isoformat()},
                {"milestone": "Closure", "target_date": (now + timedelta(days=target_days)).isoformat()}
            ]
        )

        self._remediation_plans.append(plan)
        return plan

    def calculate_compliance_score(self) -> ComplianceScorecard:
        """
        Calculate overall compliance scorecard.

        Returns:
            ComplianceScorecard with overall assessment
        """
        if not self._requirements:
            return ComplianceScorecard(
                scorecard_id=f"SC-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                overall_compliance_score=0,
                overall_coverage=0,
                executive_summary="No requirements registered."
            )

        # Calculate coverage for each requirement
        coverages = {}
        for req_id in self._requirements:
            coverage, _, _ = self.analyze_coverage(req_id)
            coverages[req_id] = coverage

        # Overall metrics
        total_reqs = len(self._requirements)
        fully_covered = sum(1 for c in coverages.values() if c >= 95)
        partially_covered = sum(1 for c in coverages.values() if 50 <= c < 95)
        not_covered = sum(1 for c in coverages.values() if c < 50)

        overall_coverage = sum(coverages.values()) / total_reqs if total_reqs > 0 else 0

        # Calculate score (coverage weighted by severity of gaps)
        base_score = overall_coverage
        critical_penalty = len([g for g in self._gaps if g.severity == GapSeverity.CRITICAL]) * 5
        high_penalty = len([g for g in self._gaps if g.severity == GapSeverity.HIGH]) * 2

        overall_score = max(0, base_score - critical_penalty - high_penalty)

        # Category scores
        category_scores = {}
        for req in self._requirements.values():
            cat = req.category
            if cat not in category_scores:
                category_scores[cat] = []
            category_scores[cat].append(coverages.get(req.requirement_id, 0))

        category_scores = {
            cat: sum(scores) / len(scores) if scores else 0
            for cat, scores in category_scores.items()
        }

        # Gap counts
        total_gaps = len(self._gaps)
        critical_gaps = len([g for g in self._gaps if g.severity == GapSeverity.CRITICAL])
        high_gaps = len([g for g in self._gaps if g.severity == GapSeverity.HIGH])

        # Remediation status
        rem_total = len(self._remediation_plans)
        rem_completed = len([p for p in self._remediation_plans if p.status == "completed"])
        rem_in_progress = len([p for p in self._remediation_plans if p.status == "in_progress"])

        # Generate summary
        key_findings = []
        if critical_gaps > 0:
            key_findings.append(f"{critical_gaps} critical gaps require immediate attention")
        if overall_coverage < 80:
            key_findings.append(f"Overall coverage at {overall_coverage:.1f}% - below target")
        if not_covered > 0:
            key_findings.append(f"{not_covered} requirements have less than 50% coverage")

        recommendations = []
        if critical_gaps > 0:
            recommendations.append("Prioritize remediation of critical gaps")
        if overall_coverage < 90:
            recommendations.append("Enhance control framework coverage")
        if rem_in_progress > 0:
            recommendations.append(f"Monitor {rem_in_progress} remediation plans in progress")

        summary = f"Compliance assessment shows {overall_score:.1f}% score with {overall_coverage:.1f}% coverage. "
        summary += f"{fully_covered} of {total_reqs} requirements fully covered. "
        if total_gaps > 0:
            summary += f"{total_gaps} gaps identified ({critical_gaps} critical, {high_gaps} high)."
        else:
            summary += "No significant gaps identified."

        return ComplianceScorecard(
            scorecard_id=f"SC-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            overall_compliance_score=round(overall_score, 1),
            overall_coverage=round(overall_coverage, 1),
            category_scores={k: round(v, 1) for k, v in category_scores.items()},
            total_requirements=total_reqs,
            fully_covered=fully_covered,
            partially_covered=partially_covered,
            not_covered=not_covered,
            total_gaps=total_gaps,
            critical_gaps=critical_gaps,
            high_gaps=high_gaps,
            remediation_plans_total=rem_total,
            remediation_in_progress=rem_in_progress,
            remediation_completed=rem_completed,
            executive_summary=summary,
            key_findings=key_findings,
            recommendations=recommendations
        )


def generate_sample_data() -> Tuple[List[RegulatoryRequirement], List[Control], List[ControlMapping]]:
    """Generate sample compliance data."""
    requirements = [
        RegulatoryRequirement(
            requirement_id="REQ-APU-001",
            regulation_id="POJK 12/2017",
            regulation_name="APU-PPT",
            article="Article 11",
            requirement_text="Financial institutions must implement Customer Due Diligence (CDD) procedures",
            category="risk",
            mandatory=True,
            effective_date=datetime(2018, 1, 1)
        ),
        RegulatoryRequirement(
            requirement_id="REQ-APU-002",
            regulation_id="POJK 12/2017",
            regulation_name="APU-PPT",
            article="Article 15",
            requirement_text="Enhanced due diligence for high-risk customers",
            category="risk",
            mandatory=True,
            effective_date=datetime(2018, 1, 1)
        ),
        RegulatoryRequirement(
            requirement_id="REQ-APU-003",
            regulation_id="POJK 12/2017",
            regulation_name="APU-PPT",
            article="Article 20",
            requirement_text="Suspicious Transaction Report (STR) filing within 3 working days",
            category="reporting",
            mandatory=True,
            effective_date=datetime(2018, 1, 1)
        ),
    ]

    controls = [
        Control(
            control_id="CTL-001",
            control_name="CDD Process",
            description="Customer due diligence during onboarding",
            control_type="preventive",
            control_category="automated",
            owner_department="Compliance",
            status=ControlStatus.IMPLEMENTED,
            effectiveness=ControlEffectiveness.EFFECTIVE
        ),
        Control(
            control_id="CTL-002",
            control_name="EDD Process",
            description="Enhanced due diligence for high-risk customers",
            control_type="preventive",
            control_category="hybrid",
            owner_department="Compliance",
            status=ControlStatus.PARTIALLY_IMPLEMENTED,
            effectiveness=ControlEffectiveness.PARTIALLY_EFFECTIVE
        ),
    ]

    mappings = [
        ControlMapping(
            mapping_id="MAP-001",
            control_id="CTL-001",
            requirement_id="REQ-APU-001",
            coverage_type="full",
            coverage_percentage=100,
            validated=True
        ),
        ControlMapping(
            mapping_id="MAP-002",
            control_id="CTL-002",
            requirement_id="REQ-APU-002",
            coverage_type="partial",
            coverage_percentage=70,
            validated=True
        ),
    ]

    return requirements, controls, mappings


# Export
__all__ = [
    # Enums
    "ControlStatus",
    "ControlEffectiveness",
    "GapSeverity",
    "RemediationPriority",
    # Models
    "RegulatoryRequirement",
    "Control",
    "ControlMapping",
    "ComplianceGap",
    "RemediationPlan",
    "ComplianceScorecard",
    # Classes
    "ComplianceGapAnalyzer",
    # Functions
    "generate_sample_data",
]
