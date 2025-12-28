"""
Macro-Financial Stress Tester Module for AURIX 2026.
Simulation of BI rate hikes and Rupiah fluctuations impact on financial portfolios.

Features:
- BI Rate shock simulation
- Rupiah depreciation impact analysis
- CAR projection under stress scenarios
- Credit risk stress testing
- Liquidity stress testing
- Integrated Basel IV / OJK stress testing framework

Formula:
Projected CAR = (Total Capital + Retained Earnings - Stress Losses) /
                (Credit RWA + Market RWA + Operational RWA)
"""

from datetime import datetime, date
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from pydantic import BaseModel, Field
import math
import logging

logger = logging.getLogger(__name__)


# ============================================
# Enums and Constants
# ============================================

class ScenarioSeverity(str, Enum):
    """Stress scenario severity levels."""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    EXTREME = "extreme"


class RiskType(str, Enum):
    """Types of financial risk."""
    CREDIT = "credit"
    MARKET = "market"
    LIQUIDITY = "liquidity"
    OPERATIONAL = "operational"
    INTEREST_RATE = "interest_rate"
    FX = "fx"


class StressOutcome(str, Enum):
    """Stress test outcome classification."""
    PASS = "pass"
    MARGINAL = "marginal"
    FAIL = "fail"
    CRITICAL = "critical"


# Regulatory thresholds (OJK)
REGULATORY_THRESHOLDS = {
    "car_minimum": 8.0,  # Minimum CAR
    "car_buffer": 2.5,  # Capital conservation buffer
    "car_warning": 12.0,  # Internal warning threshold
    "npl_maximum": 5.0,  # Maximum NPL ratio
    "ldr_minimum": 78.0,  # Minimum LDR
    "ldr_maximum": 92.0,  # Maximum LDR
    "lcr_minimum": 100.0,  # Minimum LCR
}

# BI Rate historical context
BI_RATE_BASELINE = 6.00  # Current BI7DRR (as of 2024)

# USD/IDR baseline
USDIDR_BASELINE = 15500


# ============================================
# Pydantic Models
# ============================================

class PortfolioPosition(BaseModel):
    """Financial portfolio position."""
    total_assets: float = Field(..., description="Total assets in IDR billion")
    total_capital: float = Field(..., description="Total capital in IDR billion")
    retained_earnings: float = Field(..., description="Retained earnings in IDR billion")
    credit_rwa: float = Field(..., description="Credit Risk Weighted Assets in IDR billion")
    market_rwa: float = Field(..., description="Market Risk Weighted Assets in IDR billion")
    operational_rwa: float = Field(..., description="Operational Risk Weighted Assets in IDR billion")
    total_loans: float = Field(..., description="Total loan portfolio in IDR billion")
    npl_amount: float = Field(..., description="Non-performing loans in IDR billion")
    fx_exposure: float = Field(..., description="Foreign currency exposure in USD million")
    fixed_rate_assets: float = Field(..., description="Fixed rate assets in IDR billion")
    floating_rate_assets: float = Field(..., description="Floating rate assets in IDR billion")
    fixed_rate_liabilities: float = Field(..., description="Fixed rate liabilities in IDR billion")
    floating_rate_liabilities: float = Field(..., description="Floating rate liabilities in IDR billion")
    liquid_assets: float = Field(..., description="High quality liquid assets in IDR billion")
    net_cash_outflow_30d: float = Field(..., description="30-day net cash outflow in IDR billion")

    @property
    def current_car(self) -> float:
        """Calculate current CAR."""
        total_rwa = self.credit_rwa + self.market_rwa + self.operational_rwa
        if total_rwa == 0:
            return 0
        return ((self.total_capital + self.retained_earnings) / total_rwa) * 100

    @property
    def current_npl_ratio(self) -> float:
        """Calculate current NPL ratio."""
        if self.total_loans == 0:
            return 0
        return (self.npl_amount / self.total_loans) * 100

    @property
    def current_lcr(self) -> float:
        """Calculate current LCR."""
        if self.net_cash_outflow_30d == 0:
            return 100
        return (self.liquid_assets / self.net_cash_outflow_30d) * 100


class StressScenario(BaseModel):
    """Definition of a stress scenario."""
    scenario_id: str = Field(..., description="Unique scenario identifier")
    scenario_name: str = Field(..., description="Scenario name")
    severity: ScenarioSeverity = Field(..., description="Scenario severity")
    description: str = Field(..., description="Scenario description")

    # Macro shocks
    bi_rate_change_bps: int = Field(default=0, description="BI rate change in basis points")
    usdidr_change_pct: float = Field(default=0, description="USD/IDR change percentage")
    gdp_growth_change_pct: float = Field(default=0, description="GDP growth change in percentage points")
    inflation_change_pct: float = Field(default=0, description="Inflation change in percentage points")

    # Risk parameters
    credit_loss_rate: float = Field(default=0, ge=0, le=100, description="Additional credit loss rate %")
    npl_migration_rate: float = Field(default=0, ge=0, le=100, description="NPL migration rate %")
    market_loss_rate: float = Field(default=0, ge=0, le=100, description="Market portfolio loss rate %")
    liquidity_outflow_rate: float = Field(default=0, ge=0, le=100, description="Additional liquidity outflow %")

    # Reference
    reference: str = Field(default="", description="Regulatory or historical reference")


class StressImpact(BaseModel):
    """Impact of stress on a specific risk type."""
    risk_type: RiskType = Field(..., description="Type of risk")
    impact_amount: float = Field(..., description="Impact amount in IDR billion")
    impact_percentage: float = Field(..., description="Impact as percentage of baseline")
    description: str = Field(..., description="Impact description")


class ProjectedMetrics(BaseModel):
    """Projected financial metrics under stress."""
    projected_car: float = Field(..., description="Projected CAR %")
    car_change: float = Field(..., description="CAR change in percentage points")
    projected_npl: float = Field(..., description="Projected NPL ratio %")
    npl_change: float = Field(..., description="NPL change in percentage points")
    projected_lcr: float = Field(..., description="Projected LCR %")
    lcr_change: float = Field(..., description="LCR change in percentage points")
    capital_shortfall: float = Field(..., description="Capital shortfall in IDR billion (if any)")
    liquidity_gap: float = Field(..., description="Liquidity gap in IDR billion (if any)")


class StressTestResult(BaseModel):
    """Complete stress test result."""
    test_id: str = Field(..., description="Test identifier")
    scenario: StressScenario = Field(..., description="Stress scenario applied")
    baseline_position: PortfolioPosition = Field(..., description="Baseline portfolio")

    # Baseline metrics
    baseline_car: float = Field(..., description="Baseline CAR %")
    baseline_npl: float = Field(..., description="Baseline NPL ratio %")
    baseline_lcr: float = Field(..., description="Baseline LCR %")

    # Stress impacts
    impacts: List[StressImpact] = Field(default_factory=list, description="Individual risk impacts")
    total_loss: float = Field(..., description="Total stress loss in IDR billion")

    # Projected metrics
    projections: ProjectedMetrics = Field(..., description="Projected metrics under stress")

    # Outcome
    outcome: StressOutcome = Field(..., description="Test outcome")
    breached_thresholds: List[str] = Field(default_factory=list, description="Regulatory thresholds breached")

    # Recommendations
    narrative: str = Field(..., description="AI-generated narrative")
    recommendations: List[str] = Field(default_factory=list, description="Recommended actions")

    timestamp: datetime = Field(default_factory=datetime.now)


class StressTestSuite(BaseModel):
    """Suite of stress test results."""
    suite_id: str = Field(..., description="Suite identifier")
    entity_name: str = Field(..., description="Entity name")
    baseline_position: PortfolioPosition = Field(..., description="Baseline portfolio")
    scenarios_tested: int = Field(..., description="Number of scenarios tested")
    results: List[StressTestResult] = Field(default_factory=list, description="Individual results")

    # Summary
    worst_case_car: float = Field(..., description="Worst case projected CAR")
    worst_case_scenario: str = Field(..., description="Worst case scenario name")
    pass_count: int = Field(..., description="Scenarios passed")
    fail_count: int = Field(..., description="Scenarios failed")

    # Overall assessment
    overall_resilience: str = Field(..., description="Overall resilience assessment")
    key_vulnerabilities: List[str] = Field(default_factory=list, description="Key vulnerabilities identified")

    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================
# Pre-defined Stress Scenarios
# ============================================

STRESS_SCENARIOS: Dict[str, StressScenario] = {
    "bi_rate_mild": StressScenario(
        scenario_id="BI_RATE_MILD",
        scenario_name="BI Rate Hike - Mild",
        severity=ScenarioSeverity.MILD,
        description="BI rate increases by 50 bps due to inflation pressure",
        bi_rate_change_bps=50,
        credit_loss_rate=0.3,
        npl_migration_rate=5.0,
        reference="Historical average adjustment"
    ),
    "bi_rate_moderate": StressScenario(
        scenario_id="BI_RATE_MODERATE",
        scenario_name="BI Rate Hike - Moderate",
        severity=ScenarioSeverity.MODERATE,
        description="BI rate increases by 150 bps due to Fed rate hikes and inflation",
        bi_rate_change_bps=150,
        credit_loss_rate=0.8,
        npl_migration_rate=12.0,
        market_loss_rate=3.0,
        reference="2022-2023 tightening cycle reference"
    ),
    "bi_rate_severe": StressScenario(
        scenario_id="BI_RATE_SEVERE",
        scenario_name="BI Rate Hike - Severe",
        severity=ScenarioSeverity.SEVERE,
        description="BI rate increases by 300 bps in aggressive tightening",
        bi_rate_change_bps=300,
        credit_loss_rate=2.0,
        npl_migration_rate=25.0,
        market_loss_rate=8.0,
        liquidity_outflow_rate=15.0,
        reference="1998 Crisis-level adjustment"
    ),
    "rupiah_mild": StressScenario(
        scenario_id="RUPIAH_MILD",
        scenario_name="Rupiah Depreciation - Mild",
        severity=ScenarioSeverity.MILD,
        description="IDR depreciates 5% against USD",
        usdidr_change_pct=5.0,
        market_loss_rate=1.5,
        reference="Normal volatility range"
    ),
    "rupiah_moderate": StressScenario(
        scenario_id="RUPIAH_MODERATE",
        scenario_name="Rupiah Depreciation - Moderate",
        severity=ScenarioSeverity.MODERATE,
        description="IDR depreciates 15% against USD due to capital outflows",
        usdidr_change_pct=15.0,
        bi_rate_change_bps=75,
        market_loss_rate=5.0,
        credit_loss_rate=0.5,
        npl_migration_rate=8.0,
        reference="2018 EM selloff reference"
    ),
    "rupiah_severe": StressScenario(
        scenario_id="RUPIAH_SEVERE",
        scenario_name="Rupiah Depreciation - Severe",
        severity=ScenarioSeverity.SEVERE,
        description="IDR depreciates 30% in currency crisis",
        usdidr_change_pct=30.0,
        bi_rate_change_bps=200,
        market_loss_rate=12.0,
        credit_loss_rate=3.0,
        npl_migration_rate=35.0,
        liquidity_outflow_rate=25.0,
        reference="1997-1998 Asian Financial Crisis"
    ),
    "combined_moderate": StressScenario(
        scenario_id="COMBINED_MODERATE",
        scenario_name="Combined Shock - Moderate",
        severity=ScenarioSeverity.MODERATE,
        description="Combined BI rate hike and Rupiah depreciation",
        bi_rate_change_bps=100,
        usdidr_change_pct=10.0,
        gdp_growth_change_pct=-1.0,
        credit_loss_rate=1.2,
        npl_migration_rate=15.0,
        market_loss_rate=6.0,
        liquidity_outflow_rate=10.0,
        reference="OJK Stress Test Scenario 2024"
    ),
    "combined_severe": StressScenario(
        scenario_id="COMBINED_SEVERE",
        scenario_name="Combined Shock - Severe",
        severity=ScenarioSeverity.SEVERE,
        description="Severe economic downturn with multiple shocks",
        bi_rate_change_bps=250,
        usdidr_change_pct=25.0,
        gdp_growth_change_pct=-4.0,
        inflation_change_pct=5.0,
        credit_loss_rate=4.0,
        npl_migration_rate=40.0,
        market_loss_rate=15.0,
        liquidity_outflow_rate=30.0,
        reference="Basel III Severely Adverse Scenario"
    ),
    "pandemic_shock": StressScenario(
        scenario_id="PANDEMIC_SHOCK",
        scenario_name="Pandemic Economic Shock",
        severity=ScenarioSeverity.EXTREME,
        description="Pandemic-level economic disruption",
        bi_rate_change_bps=-100,
        usdidr_change_pct=15.0,
        gdp_growth_change_pct=-5.0,
        credit_loss_rate=5.0,
        npl_migration_rate=50.0,
        market_loss_rate=20.0,
        liquidity_outflow_rate=20.0,
        reference="COVID-19 2020 reference"
    )
}


# ============================================
# Macro-Financial Stress Tester Engine
# ============================================

class MacroStressTester:
    """
    Macro-Financial Stress Testing Engine.
    Simulates impact of macroeconomic shocks on financial portfolios.
    """

    def __init__(self):
        """Initialize the stress tester."""
        self.scenarios = STRESS_SCENARIOS
        self.thresholds = REGULATORY_THRESHOLDS
        self._test_counter = 0

    def _generate_test_id(self) -> str:
        """Generate unique test ID."""
        self._test_counter += 1
        return f"STRESS-{datetime.now().strftime('%Y%m%d')}-{self._test_counter:04d}"

    def calculate_interest_rate_impact(
        self,
        position: PortfolioPosition,
        rate_change_bps: int
    ) -> StressImpact:
        """
        Calculate impact of interest rate change.

        Uses duration gap analysis simplified.
        """
        # Simplified duration gap analysis
        # Assume average duration of 2 years for fixed rate instruments
        duration = 2.0
        rate_change_decimal = rate_change_bps / 10000

        # Impact on fixed rate assets (negative when rates rise)
        fixed_rate_gap = position.fixed_rate_assets - position.fixed_rate_liabilities
        price_impact = -duration * rate_change_decimal * fixed_rate_gap

        # Impact on NIM for floating rate
        floating_gap = position.floating_rate_assets - position.floating_rate_liabilities
        nim_impact = floating_gap * rate_change_decimal * 0.5  # Assume 6-month repricing

        total_impact = abs(price_impact) + (nim_impact if nim_impact < 0 else 0)

        return StressImpact(
            risk_type=RiskType.INTEREST_RATE,
            impact_amount=round(total_impact, 2),
            impact_percentage=round((total_impact / position.total_capital) * 100, 2),
            description=f"Interest rate shock of {rate_change_bps}bps: Price impact IDR {price_impact:.1f}B, NIM impact IDR {nim_impact:.1f}B"
        )

    def calculate_fx_impact(
        self,
        position: PortfolioPosition,
        usdidr_change_pct: float
    ) -> StressImpact:
        """Calculate impact of FX rate change."""
        # Convert USD exposure to IDR at baseline rate
        exposure_idr = position.fx_exposure * USDIDR_BASELINE / 1000  # Convert to billion

        # Loss from depreciation (if net long USD liability position)
        # Assume 30% of FX exposure is unhedged
        unhedged_pct = 0.30
        fx_loss = exposure_idr * unhedged_pct * (usdidr_change_pct / 100)

        return StressImpact(
            risk_type=RiskType.FX,
            impact_amount=round(abs(fx_loss), 2),
            impact_percentage=round((abs(fx_loss) / position.total_capital) * 100, 2),
            description=f"IDR depreciation of {usdidr_change_pct}%: Unhedged FX loss IDR {fx_loss:.1f}B"
        )

    def calculate_credit_impact(
        self,
        position: PortfolioPosition,
        credit_loss_rate: float,
        npl_migration_rate: float
    ) -> Tuple[StressImpact, float]:
        """
        Calculate credit risk impact.

        Returns impact and new NPL amount.
        """
        # Direct credit loss
        direct_loss = position.total_loans * (credit_loss_rate / 100)

        # NPL migration (performing loans becoming NPL)
        performing_loans = position.total_loans - position.npl_amount
        new_npl = performing_loans * (npl_migration_rate / 100)

        # Provision requirement for new NPL (assume 50% provision rate)
        provision_loss = new_npl * 0.50

        total_credit_loss = direct_loss + provision_loss
        new_npl_amount = position.npl_amount + new_npl

        return StressImpact(
            risk_type=RiskType.CREDIT,
            impact_amount=round(total_credit_loss, 2),
            impact_percentage=round((total_credit_loss / position.total_capital) * 100, 2),
            description=f"Credit stress: Direct loss IDR {direct_loss:.1f}B, New provisions IDR {provision_loss:.1f}B"
        ), new_npl_amount

    def calculate_market_impact(
        self,
        position: PortfolioPosition,
        market_loss_rate: float
    ) -> StressImpact:
        """Calculate market risk impact on trading book."""
        # Assume market RWA represents trading book
        trading_book = position.market_rwa * 0.08  # Reverse RWA to position
        market_loss = trading_book * (market_loss_rate / 100) * 10  # Amplify for stressed VaR

        return StressImpact(
            risk_type=RiskType.MARKET,
            impact_amount=round(market_loss, 2),
            impact_percentage=round((market_loss / position.total_capital) * 100, 2),
            description=f"Market stress: {market_loss_rate}% loss on trading positions"
        )

    def calculate_liquidity_impact(
        self,
        position: PortfolioPosition,
        liquidity_outflow_rate: float
    ) -> Tuple[StressImpact, float]:
        """
        Calculate liquidity stress impact.

        Returns impact and new LCR.
        """
        # Additional outflows under stress
        additional_outflow = position.net_cash_outflow_30d * (liquidity_outflow_rate / 100)
        stressed_outflow = position.net_cash_outflow_30d + additional_outflow

        # New LCR
        new_lcr = (position.liquid_assets / stressed_outflow) * 100 if stressed_outflow > 0 else 100

        # Liquidity gap if LCR < 100%
        liquidity_gap = 0
        if new_lcr < 100:
            liquidity_gap = stressed_outflow - position.liquid_assets

        return StressImpact(
            risk_type=RiskType.LIQUIDITY,
            impact_amount=round(additional_outflow, 2),
            impact_percentage=round(liquidity_outflow_rate, 2),
            description=f"Liquidity stress: Additional outflows IDR {additional_outflow:.1f}B"
        ), new_lcr

    def determine_outcome(
        self,
        projected_car: float,
        projected_npl: float,
        projected_lcr: float
    ) -> Tuple[StressOutcome, List[str]]:
        """Determine stress test outcome and breached thresholds."""
        breaches = []

        if projected_car < self.thresholds["car_minimum"]:
            breaches.append(f"CAR below minimum ({self.thresholds['car_minimum']}%)")
        elif projected_car < self.thresholds["car_warning"]:
            breaches.append(f"CAR below warning level ({self.thresholds['car_warning']}%)")

        if projected_npl > self.thresholds["npl_maximum"]:
            breaches.append(f"NPL exceeds maximum ({self.thresholds['npl_maximum']}%)")

        if projected_lcr < self.thresholds["lcr_minimum"]:
            breaches.append(f"LCR below minimum ({self.thresholds['lcr_minimum']}%)")

        # Determine outcome
        if projected_car < self.thresholds["car_minimum"]:
            outcome = StressOutcome.CRITICAL
        elif len(breaches) >= 2:
            outcome = StressOutcome.FAIL
        elif len(breaches) == 1:
            outcome = StressOutcome.MARGINAL
        else:
            outcome = StressOutcome.PASS

        return outcome, breaches

    def generate_narrative(
        self,
        scenario: StressScenario,
        projections: ProjectedMetrics,
        outcome: StressOutcome
    ) -> str:
        """Generate narrative for stress test result."""
        severity_text = {
            ScenarioSeverity.MILD: "mild",
            ScenarioSeverity.MODERATE: "moderate",
            ScenarioSeverity.SEVERE: "severe",
            ScenarioSeverity.EXTREME: "extreme"
        }

        outcome_text = {
            StressOutcome.PASS: "The institution demonstrates adequate resilience",
            StressOutcome.MARGINAL: "The institution shows marginal resilience with some concerns",
            StressOutcome.FAIL: "The institution fails to maintain adequate buffers",
            StressOutcome.CRITICAL: "CRITICAL: The institution would breach regulatory minimums"
        }

        narrative = f"Under the {severity_text[scenario.severity]} stress scenario '{scenario.scenario_name}', "

        if scenario.bi_rate_change_bps:
            narrative += f"a {scenario.bi_rate_change_bps}bps BI rate increase "
        if scenario.usdidr_change_pct:
            narrative += f"and {scenario.usdidr_change_pct}% Rupiah depreciation "

        narrative += f"would result in CAR declining from baseline to {projections.projected_car:.2f}% "
        narrative += f"(change of {projections.car_change:+.2f}pp). "

        if projections.npl_change > 0:
            narrative += f"NPL ratio would increase to {projections.projected_npl:.2f}%. "

        narrative += outcome_text[outcome] + "."

        if projections.capital_shortfall > 0:
            narrative += f" Capital shortfall of IDR {projections.capital_shortfall:.1f}B would need to be addressed."

        return narrative

    def generate_recommendations(
        self,
        outcome: StressOutcome,
        projections: ProjectedMetrics,
        breaches: List[str]
    ) -> List[str]:
        """Generate recommendations based on stress test results."""
        recommendations = []

        if outcome == StressOutcome.CRITICAL:
            recommendations.append("IMMEDIATE: Develop Capital Recovery Plan per OJK requirements")
            recommendations.append("Engage with OJK on capital remediation timeline")
            recommendations.append("Consider capital raising options (rights issue, subordinated debt)")

        if outcome in [StressOutcome.FAIL, StressOutcome.CRITICAL]:
            recommendations.append("Review and strengthen credit risk management framework")
            recommendations.append("Increase loan loss provisions and coverage ratio")
            recommendations.append("Reduce risk-weighted asset growth")

        if projections.projected_car < 12:
            recommendations.append("Increase capital buffer through retained earnings")
            recommendations.append("Optimize RWA through portfolio rebalancing")

        if projections.npl_change > 2:
            recommendations.append("Intensify NPL collection and recovery efforts")
            recommendations.append("Review underwriting standards for high-risk segments")

        if projections.projected_lcr < 100:
            recommendations.append("Increase high-quality liquid asset holdings")
            recommendations.append("Review and diversify funding sources")
            recommendations.append("Develop contingency funding plan")

        if not recommendations:
            recommendations.append("Continue monitoring macro-financial conditions")
            recommendations.append("Maintain current capital and liquidity buffers")

        return recommendations

    def run_stress_test(
        self,
        position: PortfolioPosition,
        scenario: StressScenario
    ) -> StressTestResult:
        """
        Run stress test for a single scenario.

        Args:
            position: Current portfolio position
            scenario: Stress scenario to apply

        Returns:
            StressTestResult with complete analysis
        """
        impacts: List[StressImpact] = []
        total_loss = 0

        # Calculate baseline metrics
        baseline_car = position.current_car
        baseline_npl = position.current_npl_ratio
        baseline_lcr = position.current_lcr

        # Interest rate impact
        if scenario.bi_rate_change_bps:
            ir_impact = self.calculate_interest_rate_impact(position, scenario.bi_rate_change_bps)
            impacts.append(ir_impact)
            total_loss += ir_impact.impact_amount

        # FX impact
        if scenario.usdidr_change_pct:
            fx_impact = self.calculate_fx_impact(position, scenario.usdidr_change_pct)
            impacts.append(fx_impact)
            total_loss += fx_impact.impact_amount

        # Credit impact
        new_npl_amount = position.npl_amount
        if scenario.credit_loss_rate or scenario.npl_migration_rate:
            credit_impact, new_npl_amount = self.calculate_credit_impact(
                position, scenario.credit_loss_rate, scenario.npl_migration_rate
            )
            impacts.append(credit_impact)
            total_loss += credit_impact.impact_amount

        # Market impact
        if scenario.market_loss_rate:
            market_impact = self.calculate_market_impact(position, scenario.market_loss_rate)
            impacts.append(market_impact)
            total_loss += market_impact.impact_amount

        # Liquidity impact
        new_lcr = baseline_lcr
        if scenario.liquidity_outflow_rate:
            liq_impact, new_lcr = self.calculate_liquidity_impact(position, scenario.liquidity_outflow_rate)
            impacts.append(liq_impact)

        # Calculate projected CAR
        stressed_capital = position.total_capital + position.retained_earnings - total_loss
        total_rwa = position.credit_rwa + position.market_rwa + position.operational_rwa

        # RWA may increase under stress (credit migration)
        rwa_increase = position.credit_rwa * (scenario.npl_migration_rate / 100) * 0.5
        stressed_rwa = total_rwa + rwa_increase

        projected_car = (stressed_capital / stressed_rwa) * 100 if stressed_rwa > 0 else 0
        projected_npl = (new_npl_amount / position.total_loans) * 100 if position.total_loans > 0 else 0

        # Calculate shortfalls
        min_capital = stressed_rwa * (self.thresholds["car_minimum"] / 100)
        capital_shortfall = max(0, min_capital - stressed_capital)

        liquidity_gap = 0
        if new_lcr < 100:
            liquidity_gap = position.net_cash_outflow_30d * (1 + scenario.liquidity_outflow_rate / 100) - position.liquid_assets
            liquidity_gap = max(0, liquidity_gap)

        projections = ProjectedMetrics(
            projected_car=round(projected_car, 2),
            car_change=round(projected_car - baseline_car, 2),
            projected_npl=round(projected_npl, 2),
            npl_change=round(projected_npl - baseline_npl, 2),
            projected_lcr=round(new_lcr, 2),
            lcr_change=round(new_lcr - baseline_lcr, 2),
            capital_shortfall=round(capital_shortfall, 2),
            liquidity_gap=round(liquidity_gap, 2)
        )

        # Determine outcome
        outcome, breaches = self.determine_outcome(projected_car, projected_npl, new_lcr)

        # Generate narrative and recommendations
        narrative = self.generate_narrative(scenario, projections, outcome)
        recommendations = self.generate_recommendations(outcome, projections, breaches)

        return StressTestResult(
            test_id=self._generate_test_id(),
            scenario=scenario,
            baseline_position=position,
            baseline_car=round(baseline_car, 2),
            baseline_npl=round(baseline_npl, 2),
            baseline_lcr=round(baseline_lcr, 2),
            impacts=impacts,
            total_loss=round(total_loss, 2),
            projections=projections,
            outcome=outcome,
            breached_thresholds=breaches,
            narrative=narrative,
            recommendations=recommendations
        )

    def run_stress_test_suite(
        self,
        entity_name: str,
        position: PortfolioPosition,
        scenario_ids: Optional[List[str]] = None
    ) -> StressTestSuite:
        """
        Run multiple stress scenarios.

        Args:
            entity_name: Entity name
            position: Portfolio position
            scenario_ids: Optional list of scenario IDs to run (defaults to all)

        Returns:
            StressTestSuite with all results
        """
        if scenario_ids is None:
            scenarios_to_run = list(self.scenarios.values())
        else:
            scenarios_to_run = [self.scenarios[sid] for sid in scenario_ids if sid in self.scenarios]

        results: List[StressTestResult] = []
        worst_car = float('inf')
        worst_scenario = ""

        for scenario in scenarios_to_run:
            result = self.run_stress_test(position, scenario)
            results.append(result)

            if result.projections.projected_car < worst_car:
                worst_car = result.projections.projected_car
                worst_scenario = scenario.scenario_name

        pass_count = sum(1 for r in results if r.outcome == StressOutcome.PASS)
        fail_count = sum(1 for r in results if r.outcome in [StressOutcome.FAIL, StressOutcome.CRITICAL])

        # Overall assessment
        if fail_count == 0:
            resilience = "STRONG - Institution passes all stress scenarios"
        elif fail_count <= len(results) * 0.25:
            resilience = "ADEQUATE - Institution passes most stress scenarios"
        elif fail_count <= len(results) * 0.5:
            resilience = "WEAK - Institution fails multiple stress scenarios"
        else:
            resilience = "CRITICAL - Institution fails majority of stress scenarios"

        # Key vulnerabilities
        vulnerabilities = []
        for result in results:
            if result.outcome in [StressOutcome.FAIL, StressOutcome.CRITICAL]:
                for impact in result.impacts:
                    if impact.impact_percentage > 10:
                        vulnerabilities.append(f"{impact.risk_type.value}: {impact.description}")

        return StressTestSuite(
            suite_id=f"SUITE-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            entity_name=entity_name,
            baseline_position=position,
            scenarios_tested=len(results),
            results=results,
            worst_case_car=round(worst_car, 2),
            worst_case_scenario=worst_scenario,
            pass_count=pass_count,
            fail_count=fail_count,
            overall_resilience=resilience,
            key_vulnerabilities=list(set(vulnerabilities))[:5]
        )


def generate_sample_portfolio() -> PortfolioPosition:
    """Generate sample portfolio position for testing."""
    return PortfolioPosition(
        total_assets=150000,  # IDR 150T
        total_capital=18000,  # IDR 18T
        retained_earnings=3500,  # IDR 3.5T
        credit_rwa=95000,  # IDR 95T
        market_rwa=8000,  # IDR 8T
        operational_rwa=12000,  # IDR 12T
        total_loans=85000,  # IDR 85T
        npl_amount=2550,  # IDR 2.55T (3% NPL)
        fx_exposure=2500,  # USD 2.5B
        fixed_rate_assets=45000,
        floating_rate_assets=40000,
        fixed_rate_liabilities=50000,
        floating_rate_liabilities=35000,
        liquid_assets=25000,
        net_cash_outflow_30d=20000
    )


# Export
__all__ = [
    # Enums
    "ScenarioSeverity",
    "RiskType",
    "StressOutcome",
    # Models
    "PortfolioPosition",
    "StressScenario",
    "StressImpact",
    "ProjectedMetrics",
    "StressTestResult",
    "StressTestSuite",
    # Constants
    "STRESS_SCENARIOS",
    "REGULATORY_THRESHOLDS",
    "BI_RATE_BASELINE",
    "USDIDR_BASELINE",
    # Classes
    "MacroStressTester",
    # Functions
    "generate_sample_portfolio",
]
