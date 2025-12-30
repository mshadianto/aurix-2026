"""
IJK Benchmarking Module for AURIX 2026.
Real-time comparison of financial metrics against Indonesian Financial Services (IJK) industry averages.

Features:
- NPL/RBC/CAR benchmarking against industry averages
- Peer group comparison (BUKU 1-4, Insurance tiers)
- Historical trend analysis
- OJK SLIK / APOLO data integration ready

Data Sources:
- OJK SLIK (Sistem Layanan Informasi Keuangan)
- APOLO (Aplikasi Pelaporan Online)
- Market Data feeds
"""

from datetime import datetime, date
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from pydantic import BaseModel, Field
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# ============================================
# Enums and Constants
# ============================================

class InstitutionType(str, Enum):
    """Indonesian financial institution types."""
    BANK_BUKU1 = "bank_buku1"  # Core capital < 1T
    BANK_BUKU2 = "bank_buku2"  # Core capital 1-5T
    BANK_BUKU3 = "bank_buku3"  # Core capital 5-30T
    BANK_BUKU4 = "bank_buku4"  # Core capital > 30T
    BPR = "bpr"  # Bank Perkreditan Rakyat
    INSURANCE_LIFE = "insurance_life"
    INSURANCE_GENERAL = "insurance_general"
    MULTIFINANCE = "multifinance"
    SECURITIES = "securities"
    PENSION_FUND = "pension_fund"


class MetricCategory(str, Enum):
    """Financial metric categories."""
    CAPITAL = "capital"
    ASSET_QUALITY = "asset_quality"
    PROFITABILITY = "profitability"
    LIQUIDITY = "liquidity"
    EFFICIENCY = "efficiency"
    SOLVENCY = "solvency"


class BenchmarkStatus(str, Enum):
    """Benchmark comparison status."""
    EXCELLENT = "excellent"  # Top quartile
    GOOD = "good"  # Above average
    AVERAGE = "average"  # Within normal range
    BELOW_AVERAGE = "below_average"  # Below average
    CONCERN = "concern"  # Bottom quartile / regulatory concern


# ============================================
# Pydantic Models
# ============================================

class IndustryBenchmark(BaseModel):
    """Industry benchmark data for a specific metric."""
    metric_id: str = Field(..., description="Metric identifier")
    metric_name: str = Field(..., description="Metric display name")
    category: MetricCategory = Field(..., description="Metric category")
    institution_type: InstitutionType = Field(..., description="Institution type")
    period: str = Field(..., description="Reporting period (YYYY-MM or YYYY-QN)")

    # Statistical values
    industry_mean: float = Field(..., description="Industry average")
    industry_median: float = Field(..., description="Industry median")
    percentile_25: float = Field(..., description="25th percentile")
    percentile_75: float = Field(..., description="75th percentile")
    percentile_90: float = Field(..., description="90th percentile")
    min_value: float = Field(..., description="Minimum value in industry")
    max_value: float = Field(..., description="Maximum value in industry")

    # Regulatory thresholds
    regulatory_min: Optional[float] = Field(None, description="Regulatory minimum threshold")
    regulatory_max: Optional[float] = Field(None, description="Regulatory maximum threshold")

    # Metadata
    sample_size: int = Field(..., description="Number of institutions in sample")
    data_source: str = Field(default="OJK", description="Data source")
    last_updated: datetime = Field(default_factory=datetime.now)


class EntityMetric(BaseModel):
    """Single entity's metric value."""
    entity_id: str = Field(..., description="Entity identifier")
    entity_name: str = Field(..., description="Entity name")
    metric_id: str = Field(..., description="Metric identifier")
    value: float = Field(..., description="Metric value")
    period: str = Field(..., description="Reporting period")
    previous_value: Optional[float] = Field(None, description="Previous period value")
    yoy_change: Optional[float] = Field(None, description="Year-over-year change %")


class BenchmarkResult(BaseModel):
    """Result of benchmarking an entity against industry."""
    entity_id: str = Field(..., description="Entity identifier")
    entity_name: str = Field(..., description="Entity name")
    metric_id: str = Field(..., description="Metric identifier")
    metric_name: str = Field(..., description="Metric display name")

    # Values
    entity_value: float = Field(..., description="Entity's metric value")
    industry_mean: float = Field(..., description="Industry average")
    industry_median: float = Field(..., description="Industry median")

    # Comparison
    percentile_rank: float = Field(..., ge=0, le=100, description="Entity's percentile rank")
    deviation_from_mean: float = Field(..., description="Deviation from mean in %")
    status: BenchmarkStatus = Field(..., description="Benchmark status")

    # Regulatory compliance
    regulatory_compliant: bool = Field(default=True, description="Meets regulatory threshold")
    regulatory_gap: Optional[float] = Field(None, description="Gap to regulatory threshold")

    # Insights
    insight: str = Field(..., description="AI-generated insight")
    recommendation: Optional[str] = Field(None, description="Recommended action")

    timestamp: datetime = Field(default_factory=datetime.now)


class BenchmarkSummary(BaseModel):
    """Summary of all benchmark comparisons for an entity."""
    entity_id: str = Field(..., description="Entity identifier")
    entity_name: str = Field(..., description="Entity name")
    institution_type: InstitutionType = Field(..., description="Institution type")
    period: str = Field(..., description="Reporting period")

    # Overall scores
    overall_percentile: float = Field(..., description="Average percentile across all metrics")
    metrics_above_average: int = Field(..., description="Count of metrics above industry average")
    metrics_below_average: int = Field(..., description="Count of metrics below industry average")
    regulatory_concerns: int = Field(..., description="Count of metrics with regulatory concerns")

    # Category scores
    capital_score: Optional[float] = Field(None, description="Capital adequacy score")
    asset_quality_score: Optional[float] = Field(None, description="Asset quality score")
    profitability_score: Optional[float] = Field(None, description="Profitability score")
    liquidity_score: Optional[float] = Field(None, description="Liquidity score")

    # Details
    results: List[BenchmarkResult] = Field(default_factory=list, description="Individual metric results")
    strengths: List[str] = Field(default_factory=list, description="Top performing areas")
    weaknesses: List[str] = Field(default_factory=list, description="Areas needing improvement")

    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================
# Industry Benchmark Data (Indonesian Banking 2024)
# ============================================

# Banking industry benchmarks based on OJK data
BANKING_BENCHMARKS_2024: Dict[str, Dict[str, Any]] = {
    "car": {
        "metric_name": "Capital Adequacy Ratio (CAR)",
        "category": MetricCategory.CAPITAL,
        "buku4": {"mean": 24.5, "median": 23.8, "p25": 21.0, "p75": 27.5, "p90": 30.0, "min": 18.5, "max": 35.0},
        "buku3": {"mean": 22.0, "median": 21.5, "p25": 18.5, "p75": 25.0, "p90": 28.0, "min": 14.0, "max": 32.0},
        "buku2": {"mean": 20.5, "median": 19.8, "p25": 16.0, "p75": 23.0, "p90": 26.0, "min": 12.0, "max": 30.0},
        "buku1": {"mean": 25.0, "median": 24.0, "p25": 18.0, "p75": 30.0, "p90": 35.0, "min": 10.0, "max": 45.0},
        "regulatory_min": 8.0,
        "unit": "%",
        "good_direction": "higher"
    },
    "npl_gross": {
        "metric_name": "NPL Ratio (Gross)",
        "category": MetricCategory.ASSET_QUALITY,
        "buku4": {"mean": 2.1, "median": 1.9, "p25": 1.2, "p75": 2.8, "p90": 3.5, "min": 0.5, "max": 4.5},
        "buku3": {"mean": 2.5, "median": 2.3, "p25": 1.5, "p75": 3.2, "p90": 4.0, "min": 0.8, "max": 5.5},
        "buku2": {"mean": 3.0, "median": 2.8, "p25": 1.8, "p75": 3.8, "p90": 4.8, "min": 1.0, "max": 6.0},
        "buku1": {"mean": 3.5, "median": 3.2, "p25": 2.0, "p75": 4.5, "p90": 5.5, "min": 0.5, "max": 8.0},
        "regulatory_max": 5.0,
        "unit": "%",
        "good_direction": "lower"
    },
    "ldr": {
        "metric_name": "Loan to Deposit Ratio (LDR)",
        "category": MetricCategory.LIQUIDITY,
        "buku4": {"mean": 85.0, "median": 86.0, "p25": 78.0, "p75": 92.0, "p90": 95.0, "min": 65.0, "max": 98.0},
        "buku3": {"mean": 88.0, "median": 89.0, "p25": 82.0, "p75": 94.0, "p90": 97.0, "min": 70.0, "max": 100.0},
        "buku2": {"mean": 82.0, "median": 83.0, "p25": 75.0, "p75": 90.0, "p90": 95.0, "min": 60.0, "max": 102.0},
        "buku1": {"mean": 78.0, "median": 80.0, "p25": 68.0, "p75": 88.0, "p90": 95.0, "min": 50.0, "max": 105.0},
        "regulatory_min": 78.0,
        "regulatory_max": 92.0,
        "unit": "%",
        "good_direction": "optimal"
    },
    "roa": {
        "metric_name": "Return on Assets (ROA)",
        "category": MetricCategory.PROFITABILITY,
        "buku4": {"mean": 2.8, "median": 2.7, "p25": 2.0, "p75": 3.5, "p90": 4.0, "min": 0.5, "max": 5.0},
        "buku3": {"mean": 2.2, "median": 2.1, "p25": 1.5, "p75": 2.8, "p90": 3.5, "min": 0.2, "max": 4.5},
        "buku2": {"mean": 1.8, "median": 1.7, "p25": 1.0, "p75": 2.4, "p90": 3.0, "min": 0.1, "max": 4.0},
        "buku1": {"mean": 1.5, "median": 1.4, "p25": 0.8, "p75": 2.0, "p90": 2.8, "min": -0.5, "max": 4.0},
        "regulatory_min": 1.25,
        "unit": "%",
        "good_direction": "higher"
    },
    "roe": {
        "metric_name": "Return on Equity (ROE)",
        "category": MetricCategory.PROFITABILITY,
        "buku4": {"mean": 18.0, "median": 17.5, "p25": 14.0, "p75": 22.0, "p90": 25.0, "min": 5.0, "max": 30.0},
        "buku3": {"mean": 14.0, "median": 13.5, "p25": 10.0, "p75": 18.0, "p90": 22.0, "min": 2.0, "max": 26.0},
        "buku2": {"mean": 11.0, "median": 10.5, "p25": 7.0, "p75": 14.0, "p90": 18.0, "min": 0.0, "max": 22.0},
        "buku1": {"mean": 9.0, "median": 8.5, "p25": 5.0, "p75": 12.0, "p90": 16.0, "min": -2.0, "max": 20.0},
        "unit": "%",
        "good_direction": "higher"
    },
    "nim": {
        "metric_name": "Net Interest Margin (NIM)",
        "category": MetricCategory.PROFITABILITY,
        "buku4": {"mean": 5.2, "median": 5.0, "p25": 4.2, "p75": 6.0, "p90": 6.8, "min": 3.0, "max": 8.0},
        "buku3": {"mean": 4.8, "median": 4.6, "p25": 3.8, "p75": 5.5, "p90": 6.2, "min": 2.5, "max": 7.5},
        "buku2": {"mean": 5.5, "median": 5.3, "p25": 4.5, "p75": 6.5, "p90": 7.5, "min": 3.0, "max": 9.0},
        "buku1": {"mean": 6.0, "median": 5.8, "p25": 4.8, "p75": 7.2, "p90": 8.5, "min": 3.0, "max": 12.0},
        "unit": "%",
        "good_direction": "higher"
    },
    "bopo": {
        "metric_name": "Operating Expense Ratio (BOPO)",
        "category": MetricCategory.EFFICIENCY,
        "buku4": {"mean": 75.0, "median": 74.0, "p25": 68.0, "p75": 82.0, "p90": 88.0, "min": 55.0, "max": 92.0},
        "buku3": {"mean": 82.0, "median": 81.0, "p25": 75.0, "p75": 88.0, "p90": 92.0, "min": 65.0, "max": 95.0},
        "buku2": {"mean": 85.0, "median": 84.0, "p25": 78.0, "p75": 90.0, "p90": 94.0, "min": 70.0, "max": 98.0},
        "buku1": {"mean": 88.0, "median": 87.0, "p25": 82.0, "p75": 93.0, "p90": 96.0, "min": 72.0, "max": 99.0},
        "regulatory_max": 85.0,
        "unit": "%",
        "good_direction": "lower"
    },
    "casa_ratio": {
        "metric_name": "CASA Ratio",
        "category": MetricCategory.LIQUIDITY,
        "buku4": {"mean": 65.0, "median": 66.0, "p25": 55.0, "p75": 75.0, "p90": 80.0, "min": 40.0, "max": 85.0},
        "buku3": {"mean": 55.0, "median": 54.0, "p25": 45.0, "p75": 65.0, "p90": 72.0, "min": 30.0, "max": 78.0},
        "buku2": {"mean": 45.0, "median": 44.0, "p25": 35.0, "p75": 55.0, "p90": 62.0, "min": 25.0, "max": 70.0},
        "buku1": {"mean": 40.0, "median": 38.0, "p25": 30.0, "p75": 50.0, "p90": 58.0, "min": 20.0, "max": 65.0},
        "unit": "%",
        "good_direction": "higher"
    }
}

# Insurance industry benchmarks
INSURANCE_BENCHMARKS_2024: Dict[str, Dict[str, Any]] = {
    "rbc": {
        "metric_name": "Risk Based Capital (RBC)",
        "category": MetricCategory.SOLVENCY,
        "life": {"mean": 450.0, "median": 420.0, "p25": 280.0, "p75": 550.0, "p90": 700.0, "min": 120.0, "max": 1200.0},
        "general": {"mean": 380.0, "median": 350.0, "p25": 250.0, "p75": 480.0, "p90": 600.0, "min": 120.0, "max": 900.0},
        "regulatory_min": 120.0,
        "unit": "%",
        "good_direction": "higher"
    },
    "loss_ratio": {
        "metric_name": "Loss Ratio",
        "category": MetricCategory.EFFICIENCY,
        "life": {"mean": 55.0, "median": 54.0, "p25": 45.0, "p75": 65.0, "p90": 75.0, "min": 30.0, "max": 85.0},
        "general": {"mean": 50.0, "median": 48.0, "p25": 40.0, "p75": 60.0, "p90": 70.0, "min": 25.0, "max": 80.0},
        "unit": "%",
        "good_direction": "lower"
    },
    "expense_ratio": {
        "metric_name": "Expense Ratio",
        "category": MetricCategory.EFFICIENCY,
        "life": {"mean": 25.0, "median": 24.0, "p25": 18.0, "p75": 32.0, "p90": 38.0, "min": 10.0, "max": 45.0},
        "general": {"mean": 35.0, "median": 34.0, "p25": 28.0, "p75": 42.0, "p90": 48.0, "min": 18.0, "max": 55.0},
        "unit": "%",
        "good_direction": "lower"
    }
}


# ============================================
# IJK Benchmarking Engine
# ============================================

class IJKBenchmarkEngine:
    """
    IJK Benchmarking Engine for Indonesian Financial Services.
    Compares entity metrics against industry averages.
    """

    def __init__(self):
        """Initialize the benchmark engine."""
        self.banking_benchmarks = BANKING_BENCHMARKS_2024
        self.insurance_benchmarks = INSURANCE_BENCHMARKS_2024

    def get_benchmark_data(
        self,
        metric_id: str,
        institution_type: InstitutionType,
        period: str = "2024-Q4"
    ) -> Optional[IndustryBenchmark]:
        """
        Get industry benchmark data for a specific metric.

        Args:
            metric_id: Metric identifier (e.g., 'car', 'npl_gross')
            institution_type: Type of financial institution
            period: Reporting period

        Returns:
            IndustryBenchmark object or None if not found
        """
        # Determine which benchmark set to use
        if institution_type in [InstitutionType.INSURANCE_LIFE, InstitutionType.INSURANCE_GENERAL]:
            benchmarks = self.insurance_benchmarks
            type_key = "life" if institution_type == InstitutionType.INSURANCE_LIFE else "general"
        else:
            benchmarks = self.banking_benchmarks
            type_key = institution_type.value.replace("bank_", "")

        if metric_id not in benchmarks:
            return None

        metric_data = benchmarks[metric_id]
        if type_key not in metric_data:
            return None

        stats = metric_data[type_key]

        return IndustryBenchmark(
            metric_id=metric_id,
            metric_name=metric_data["metric_name"],
            category=metric_data["category"],
            institution_type=institution_type,
            period=period,
            industry_mean=stats["mean"],
            industry_median=stats["median"],
            percentile_25=stats["p25"],
            percentile_75=stats["p75"],
            percentile_90=stats["p90"],
            min_value=stats["min"],
            max_value=stats["max"],
            regulatory_min=metric_data.get("regulatory_min"),
            regulatory_max=metric_data.get("regulatory_max"),
            sample_size=50 if "buku" in type_key else 30,
            data_source="OJK Statistics 2024"
        )

    def calculate_percentile_rank(
        self,
        value: float,
        benchmark: IndustryBenchmark,
        good_direction: str = "higher"
    ) -> float:
        """
        Calculate the percentile rank of a value within industry distribution.

        Uses linear interpolation between known percentiles.
        """
        # Create percentile mapping
        percentiles = [
            (0, benchmark.min_value),
            (25, benchmark.percentile_25),
            (50, benchmark.industry_median),
            (75, benchmark.percentile_75),
            (90, benchmark.percentile_90),
            (100, benchmark.max_value)
        ]

        # Find position
        for i in range(len(percentiles) - 1):
            p1, v1 = percentiles[i]
            p2, v2 = percentiles[i + 1]

            if v1 <= value <= v2:
                # Linear interpolation
                if v2 == v1:
                    return (p1 + p2) / 2
                ratio = (value - v1) / (v2 - v1)
                return p1 + ratio * (p2 - p1)

        # Value outside range
        if value < benchmark.min_value:
            return 0.0
        return 100.0

    def determine_status(
        self,
        percentile: float,
        regulatory_compliant: bool,
        good_direction: str
    ) -> BenchmarkStatus:
        """Determine benchmark status based on percentile and compliance."""
        if not regulatory_compliant:
            return BenchmarkStatus.CONCERN

        # For metrics where lower is better, invert percentile
        effective_percentile = percentile if good_direction == "higher" else (100 - percentile)

        if effective_percentile >= 75:
            return BenchmarkStatus.EXCELLENT
        elif effective_percentile >= 50:
            return BenchmarkStatus.GOOD
        elif effective_percentile >= 25:
            return BenchmarkStatus.AVERAGE
        elif effective_percentile >= 10:
            return BenchmarkStatus.BELOW_AVERAGE
        else:
            return BenchmarkStatus.CONCERN

    def generate_insight(
        self,
        metric_name: str,
        value: float,
        benchmark: IndustryBenchmark,
        status: BenchmarkStatus,
        good_direction: str
    ) -> Tuple[str, Optional[str]]:
        """Generate insight and recommendation for a metric."""
        deviation = ((value - benchmark.industry_mean) / benchmark.industry_mean) * 100

        if status == BenchmarkStatus.EXCELLENT:
            insight = f"{metric_name} at {value:.2f}% is in the top quartile, {abs(deviation):.1f}% {'above' if deviation > 0 else 'below'} industry average."
            recommendation = None
        elif status == BenchmarkStatus.GOOD:
            insight = f"{metric_name} at {value:.2f}% performs above industry average ({benchmark.industry_mean:.2f}%)."
            recommendation = None
        elif status == BenchmarkStatus.AVERAGE:
            insight = f"{metric_name} at {value:.2f}% is within normal industry range (mean: {benchmark.industry_mean:.2f}%)."
            recommendation = f"Consider strategies to improve {metric_name} toward top quartile ({benchmark.percentile_75:.2f}%)."
        elif status == BenchmarkStatus.BELOW_AVERAGE:
            insight = f"{metric_name} at {value:.2f}% is below industry average ({benchmark.industry_mean:.2f}%)."
            if good_direction == "higher":
                recommendation = f"Priority action needed to improve {metric_name}. Industry median is {benchmark.industry_median:.2f}%."
            else:
                recommendation = f"Priority action needed to reduce {metric_name}. Industry median is {benchmark.industry_median:.2f}%."
        else:  # CONCERN
            insight = f"{metric_name} at {value:.2f}% requires immediate attention - regulatory threshold at risk."
            if benchmark.regulatory_min:
                recommendation = f"URGENT: Improve {metric_name} above regulatory minimum of {benchmark.regulatory_min:.2f}%."
            elif benchmark.regulatory_max:
                recommendation = f"URGENT: Reduce {metric_name} below regulatory maximum of {benchmark.regulatory_max:.2f}%."
            else:
                recommendation = f"URGENT: Review and improve {metric_name} performance."

        return insight, recommendation

    def benchmark_metric(
        self,
        entity_metric: EntityMetric,
        institution_type: InstitutionType
    ) -> Optional[BenchmarkResult]:
        """
        Benchmark a single metric against industry.

        Args:
            entity_metric: Entity's metric data
            institution_type: Type of financial institution

        Returns:
            BenchmarkResult or None if benchmark data unavailable
        """
        benchmark = self.get_benchmark_data(
            entity_metric.metric_id,
            institution_type,
            entity_metric.period
        )

        if not benchmark:
            return None

        # Get metric metadata
        if institution_type in [InstitutionType.INSURANCE_LIFE, InstitutionType.INSURANCE_GENERAL]:
            metric_data = self.insurance_benchmarks.get(entity_metric.metric_id, {})
        else:
            metric_data = self.banking_benchmarks.get(entity_metric.metric_id, {})

        good_direction = metric_data.get("good_direction", "higher")

        # Calculate percentile rank
        percentile = self.calculate_percentile_rank(
            entity_metric.value,
            benchmark,
            good_direction
        )

        # Check regulatory compliance
        regulatory_compliant = True
        regulatory_gap = None

        if benchmark.regulatory_min and entity_metric.value < benchmark.regulatory_min:
            regulatory_compliant = False
            regulatory_gap = benchmark.regulatory_min - entity_metric.value
        elif benchmark.regulatory_max and entity_metric.value > benchmark.regulatory_max:
            regulatory_compliant = False
            regulatory_gap = entity_metric.value - benchmark.regulatory_max

        # Determine status
        status = self.determine_status(percentile, regulatory_compliant, good_direction)

        # Calculate deviation from mean
        deviation = ((entity_metric.value - benchmark.industry_mean) / benchmark.industry_mean) * 100

        # Generate insights
        insight, recommendation = self.generate_insight(
            benchmark.metric_name,
            entity_metric.value,
            benchmark,
            status,
            good_direction
        )

        return BenchmarkResult(
            entity_id=entity_metric.entity_id,
            entity_name=entity_metric.entity_name,
            metric_id=entity_metric.metric_id,
            metric_name=benchmark.metric_name,
            entity_value=entity_metric.value,
            industry_mean=benchmark.industry_mean,
            industry_median=benchmark.industry_median,
            percentile_rank=round(percentile, 1),
            deviation_from_mean=round(deviation, 2),
            status=status,
            regulatory_compliant=regulatory_compliant,
            regulatory_gap=regulatory_gap,
            insight=insight,
            recommendation=recommendation
        )

    def benchmark_entity(
        self,
        entity_id: str,
        entity_name: str,
        institution_type: InstitutionType,
        metrics: Dict[str, float],
        period: str = "2024-Q4"
    ) -> BenchmarkSummary:
        """
        Comprehensive benchmark of an entity against industry.

        Args:
            entity_id: Entity identifier
            entity_name: Entity name
            institution_type: Type of financial institution
            metrics: Dict of metric_id -> value
            period: Reporting period

        Returns:
            BenchmarkSummary with all results and insights
        """
        results: List[BenchmarkResult] = []
        category_scores: Dict[MetricCategory, List[float]] = {cat: [] for cat in MetricCategory}

        for metric_id, value in metrics.items():
            entity_metric = EntityMetric(
                entity_id=entity_id,
                entity_name=entity_name,
                metric_id=metric_id,
                value=value,
                period=period
            )

            result = self.benchmark_metric(entity_metric, institution_type)
            if result:
                results.append(result)

                # Get category for scoring
                if institution_type in [InstitutionType.INSURANCE_LIFE, InstitutionType.INSURANCE_GENERAL]:
                    metric_data = self.insurance_benchmarks.get(metric_id, {})
                else:
                    metric_data = self.banking_benchmarks.get(metric_id, {})

                category = metric_data.get("category")
                if category:
                    category_scores[category].append(result.percentile_rank)

        # Calculate summary statistics
        all_percentiles = [r.percentile_rank for r in results]
        overall_percentile = sum(all_percentiles) / len(all_percentiles) if all_percentiles else 50.0

        above_avg = sum(1 for r in results if r.status in [BenchmarkStatus.EXCELLENT, BenchmarkStatus.GOOD])
        below_avg = sum(1 for r in results if r.status in [BenchmarkStatus.BELOW_AVERAGE, BenchmarkStatus.CONCERN])
        concerns = sum(1 for r in results if not r.regulatory_compliant)

        # Calculate category scores
        capital_score = sum(category_scores[MetricCategory.CAPITAL]) / len(category_scores[MetricCategory.CAPITAL]) if category_scores[MetricCategory.CAPITAL] else None
        asset_score = sum(category_scores[MetricCategory.ASSET_QUALITY]) / len(category_scores[MetricCategory.ASSET_QUALITY]) if category_scores[MetricCategory.ASSET_QUALITY] else None
        profit_score = sum(category_scores[MetricCategory.PROFITABILITY]) / len(category_scores[MetricCategory.PROFITABILITY]) if category_scores[MetricCategory.PROFITABILITY] else None
        liquid_score = sum(category_scores[MetricCategory.LIQUIDITY]) / len(category_scores[MetricCategory.LIQUIDITY]) if category_scores[MetricCategory.LIQUIDITY] else None

        # Identify strengths and weaknesses
        sorted_results = sorted(results, key=lambda r: r.percentile_rank, reverse=True)
        strengths = [f"{r.metric_name} (P{r.percentile_rank:.0f})" for r in sorted_results[:3] if r.percentile_rank >= 50]
        weaknesses = [f"{r.metric_name} (P{r.percentile_rank:.0f})" for r in sorted_results[-3:] if r.percentile_rank < 50]

        return BenchmarkSummary(
            entity_id=entity_id,
            entity_name=entity_name,
            institution_type=institution_type,
            period=period,
            overall_percentile=round(overall_percentile, 1),
            metrics_above_average=above_avg,
            metrics_below_average=below_avg,
            regulatory_concerns=concerns,
            capital_score=round(capital_score, 1) if capital_score else None,
            asset_quality_score=round(asset_score, 1) if asset_score else None,
            profitability_score=round(profit_score, 1) if profit_score else None,
            liquidity_score=round(liquid_score, 1) if liquid_score else None,
            results=results,
            strengths=strengths,
            weaknesses=weaknesses
        )


def generate_sample_entity_metrics() -> Dict[str, float]:
    """Generate sample banking metrics for testing."""
    return {
        "car": 18.5,
        "npl_gross": 3.2,
        "ldr": 88.0,
        "roa": 1.9,
        "roe": 12.5,
        "nim": 4.8,
        "bopo": 82.0,
        "casa_ratio": 52.0
    }


# ============================================
# Historical Trend Models
# ============================================

class HistoricalDataPoint(BaseModel):
    """Single historical data point."""
    period: str = Field(..., description="Period (YYYY-MM or YYYY-QN)")
    value: float = Field(..., description="Metric value")
    industry_mean: float = Field(..., description="Industry mean at that time")
    percentile_rank: float = Field(..., description="Percentile rank at that time")


class HistoricalTrend(BaseModel):
    """Historical trend data for a metric."""
    metric_id: str = Field(..., description="Metric identifier")
    metric_name: str = Field(..., description="Metric display name")
    entity_id: str = Field(..., description="Entity identifier")
    institution_type: InstitutionType = Field(..., description="Institution type")

    # Historical data points (12 months)
    data_points: List[HistoricalDataPoint] = Field(default_factory=list)

    # Trend analysis
    trend_direction: str = Field(..., description="up, down, stable")
    trend_strength: float = Field(..., description="Trend strength 0-1")
    avg_percentile: float = Field(..., description="Average percentile over period")
    percentile_change: float = Field(..., description="Change in percentile rank")

    # Projections
    projected_next_period: Optional[float] = Field(None, description="Projected value for next period")
    confidence_interval: Optional[Tuple[float, float]] = Field(None, description="95% CI for projection")


class PeerRanking(BaseModel):
    """Peer comparison ranking result."""
    entity_id: str = Field(..., description="Entity identifier")
    entity_name: str = Field(..., description="Entity name")
    metric_id: str = Field(..., description="Metric identifier")
    metric_name: str = Field(..., description="Metric display name")

    # Ranking
    rank: int = Field(..., description="Rank among peers (1 = best)")
    total_peers: int = Field(..., description="Total number of peers")
    percentile: float = Field(..., description="Percentile position")

    # Values
    entity_value: float = Field(..., description="Entity's value")
    peer_values: List[Tuple[str, float]] = Field(default_factory=list, description="Peer name and values")

    # Statistics
    peer_mean: float = Field(..., description="Peer group mean")
    peer_median: float = Field(..., description="Peer group median")
    peer_best: float = Field(..., description="Best peer value")
    peer_worst: float = Field(..., description="Worst peer value")

    # Gap analysis
    gap_to_best: float = Field(..., description="Gap to best performer")
    gap_to_median: float = Field(..., description="Gap to median")

    narrative: str = Field(..., description="Ranking narrative")


class PeerComparisonSummary(BaseModel):
    """Summary of peer comparison across all metrics."""
    entity_id: str = Field(..., description="Entity identifier")
    entity_name: str = Field(..., description="Entity name")
    peer_group: str = Field(..., description="Peer group name")
    total_peers: int = Field(..., description="Total peers in group")

    # Overall ranking
    overall_rank: int = Field(..., description="Overall composite rank")
    avg_percentile: float = Field(..., description="Average percentile across metrics")
    metrics_in_top_quartile: int = Field(..., description="Metrics in top 25%")
    metrics_in_bottom_quartile: int = Field(..., description="Metrics in bottom 25%")

    # Individual rankings
    rankings: List[PeerRanking] = Field(default_factory=list)

    # Competitive position
    competitive_strengths: List[str] = Field(default_factory=list)
    competitive_weaknesses: List[str] = Field(default_factory=list)
    key_differentiators: List[str] = Field(default_factory=list)

    narrative: str = Field(..., description="Summary narrative")
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================
# OJK Data Fetcher (Mock Implementation)
# ============================================

class OJKDataFetcher:
    """
    OJK Data Fetcher for live industry statistics.
    Note: This is a mock implementation. Real implementation would connect to OJK SLIK/APOLO.
    """

    def __init__(self, cache_ttl_hours: int = 24):
        """Initialize with cache settings."""
        self.cache_ttl_hours = cache_ttl_hours
        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self._historical_cache: Dict[str, List[HistoricalDataPoint]] = {}

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self._cache:
            return False
        cached_time, _ = self._cache[key]
        age_hours = (datetime.now() - cached_time).total_seconds() / 3600
        return age_hours < self.cache_ttl_hours

    def fetch_industry_statistics(
        self,
        metric_id: str,
        institution_type: InstitutionType,
        period: str
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch industry statistics from OJK.
        In production, this would call OJK API.

        Returns dict with mean, median, percentiles, etc.
        """
        cache_key = f"{metric_id}_{institution_type.value}_{period}"

        if self._is_cache_valid(cache_key):
            _, data = self._cache[cache_key]
            return data

        # Mock: Use static benchmarks with slight random variation
        import random

        base_benchmarks = BANKING_BENCHMARKS_2024 if "bank" in institution_type.value else INSURANCE_BENCHMARKS_2024

        if metric_id not in base_benchmarks:
            return None

        metric_data = base_benchmarks[metric_id]
        type_key = institution_type.value.replace("bank_", "").replace("insurance_", "")

        if type_key not in metric_data:
            return None

        base = metric_data[type_key]

        # Add slight variation to simulate live data
        variation = random.uniform(0.95, 1.05)

        stats = {
            "mean": round(base["mean"] * variation, 2),
            "median": round(base["median"] * variation, 2),
            "p25": round(base["p25"] * variation, 2),
            "p75": round(base["p75"] * variation, 2),
            "p90": round(base["p90"] * variation, 2),
            "min": round(base["min"] * variation, 2),
            "max": round(base["max"] * variation, 2),
            "sample_size": random.randint(45, 55),
            "last_updated": datetime.now().isoformat()
        }

        self._cache[cache_key] = (datetime.now(), stats)
        return stats

    def fetch_peer_data(
        self,
        institution_type: InstitutionType,
        metric_id: str,
        period: str
    ) -> List[Tuple[str, float]]:
        """
        Fetch peer institution data for comparison.
        Returns list of (peer_name, value) tuples.
        """
        import random

        # Generate mock peer data
        peer_count = random.randint(8, 15)
        base_benchmarks = BANKING_BENCHMARKS_2024 if "bank" in institution_type.value else INSURANCE_BENCHMARKS_2024

        if metric_id not in base_benchmarks:
            return []

        metric_data = base_benchmarks[metric_id]
        type_key = institution_type.value.replace("bank_", "").replace("insurance_", "")

        if type_key not in metric_data:
            return []

        base = metric_data[type_key]

        # Generate peer names and values
        peer_prefixes = ["Bank", "PT", "BPD", "Bank Syariah"]
        peer_suffixes = ["Mandiri", "Central Asia", "Negara Indonesia", "Rakyat Indonesia",
                        "Danamon", "CIMB Niaga", "Panin", "Maybank", "OCBC NISP",
                        "Permata", "BTPN", "Mega", "Bukopin", "Sinarmas", "Jatim"]

        peers = []
        for i in range(peer_count):
            name = f"{random.choice(peer_prefixes)} {random.choice(peer_suffixes)}"
            # Generate value within industry range
            value = random.uniform(base["min"], base["max"])
            peers.append((name, round(value, 2)))

        return sorted(peers, key=lambda x: x[1], reverse=(metric_data.get("good_direction", "higher") == "higher"))

    def fetch_historical_data(
        self,
        entity_id: str,
        metric_id: str,
        months: int = 12
    ) -> List[HistoricalDataPoint]:
        """
        Fetch historical data for trend analysis.
        Returns 12 months of historical data points.
        """
        import random
        from datetime import timedelta

        cache_key = f"hist_{entity_id}_{metric_id}"

        if cache_key in self._historical_cache:
            return self._historical_cache[cache_key]

        # Generate mock historical data
        data_points = []
        current_date = datetime.now()

        # Get base benchmark for realistic values
        base_value = 20.0  # Default
        if metric_id in BANKING_BENCHMARKS_2024:
            metric_data = BANKING_BENCHMARKS_2024[metric_id]
            if "buku3" in metric_data:
                base_value = metric_data["buku3"]["mean"]

        # Generate trend
        trend = random.choice(["improving", "declining", "stable"])
        trend_factor = 0.02 if trend == "improving" else -0.02 if trend == "declining" else 0

        for i in range(months - 1, -1, -1):
            period_date = current_date - timedelta(days=30 * i)
            period = period_date.strftime("%Y-%m")

            # Calculate value with trend and noise
            time_effect = trend_factor * (months - i - 1)
            noise = random.uniform(-0.05, 0.05)
            value = base_value * (1 + time_effect + noise)

            # Industry mean varies slightly
            industry_mean = base_value * (1 + random.uniform(-0.03, 0.03))

            # Calculate percentile
            percentile = 50 + (value - industry_mean) / industry_mean * 100

            data_points.append(HistoricalDataPoint(
                period=period,
                value=round(value, 2),
                industry_mean=round(industry_mean, 2),
                percentile_rank=round(min(max(percentile, 0), 100), 1)
            ))

        self._historical_cache[cache_key] = data_points
        return data_points


# ============================================
# Historical Trend Analyzer
# ============================================

class HistoricalTrendAnalyzer:
    """Analyzes historical trends for benchmarking metrics."""

    def __init__(self, data_fetcher: Optional[OJKDataFetcher] = None):
        """Initialize with data fetcher."""
        self.data_fetcher = data_fetcher or OJKDataFetcher()

    def analyze_trend(
        self,
        entity_id: str,
        entity_name: str,
        metric_id: str,
        metric_name: str,
        institution_type: InstitutionType,
        months: int = 12
    ) -> HistoricalTrend:
        """
        Analyze historical trend for a metric.

        Args:
            entity_id: Entity identifier
            entity_name: Entity name
            metric_id: Metric to analyze
            metric_name: Display name
            institution_type: Institution type
            months: Number of months to analyze

        Returns:
            HistoricalTrend with analysis results
        """
        import numpy as np

        # Fetch historical data
        data_points = self.data_fetcher.fetch_historical_data(entity_id, metric_id, months)

        if len(data_points) < 3:
            # Not enough data for trend analysis
            return HistoricalTrend(
                metric_id=metric_id,
                metric_name=metric_name,
                entity_id=entity_id,
                institution_type=institution_type,
                data_points=data_points,
                trend_direction="stable",
                trend_strength=0.0,
                avg_percentile=50.0,
                percentile_change=0.0
            )

        values = [dp.value for dp in data_points]
        percentiles = [dp.percentile_rank for dp in data_points]

        # Calculate trend using linear regression
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)

        # Determine trend direction and strength
        value_range = max(values) - min(values) if max(values) != min(values) else 1
        trend_strength = abs(slope * len(values)) / value_range

        if slope > 0.01 * np.mean(values):
            trend_direction = "up"
        elif slope < -0.01 * np.mean(values):
            trend_direction = "down"
        else:
            trend_direction = "stable"

        # Calculate averages and changes
        avg_percentile = np.mean(percentiles)
        percentile_change = percentiles[-1] - percentiles[0] if percentiles else 0

        # Project next period
        projected_value = slope * len(values) + intercept

        # Simple confidence interval (using standard error)
        residuals = values - (slope * x + intercept)
        std_error = np.std(residuals)
        confidence_interval = (
            round(projected_value - 1.96 * std_error, 2),
            round(projected_value + 1.96 * std_error, 2)
        )

        return HistoricalTrend(
            metric_id=metric_id,
            metric_name=metric_name,
            entity_id=entity_id,
            institution_type=institution_type,
            data_points=data_points,
            trend_direction=trend_direction,
            trend_strength=round(min(trend_strength, 1.0), 2),
            avg_percentile=round(avg_percentile, 1),
            percentile_change=round(percentile_change, 1),
            projected_next_period=round(projected_value, 2),
            confidence_interval=confidence_interval
        )


# ============================================
# Peer Ranking Engine
# ============================================

class PeerRankingEngine:
    """Engine for peer comparison and ranking."""

    def __init__(self, data_fetcher: Optional[OJKDataFetcher] = None):
        """Initialize with data fetcher."""
        self.data_fetcher = data_fetcher or OJKDataFetcher()

    def rank_against_peers(
        self,
        entity_id: str,
        entity_name: str,
        entity_value: float,
        metric_id: str,
        metric_name: str,
        institution_type: InstitutionType,
        period: str = "2024-Q4"
    ) -> PeerRanking:
        """
        Rank entity against peers for a specific metric.

        Args:
            entity_id: Entity identifier
            entity_name: Entity name
            entity_value: Entity's metric value
            metric_id: Metric to rank
            metric_name: Metric display name
            institution_type: Institution type
            period: Reporting period

        Returns:
            PeerRanking with detailed comparison
        """
        # Fetch peer data
        peer_data = self.data_fetcher.fetch_peer_data(institution_type, metric_id, period)

        if not peer_data:
            return PeerRanking(
                entity_id=entity_id,
                entity_name=entity_name,
                metric_id=metric_id,
                metric_name=metric_name,
                rank=1,
                total_peers=1,
                percentile=50.0,
                entity_value=entity_value,
                peer_values=[],
                peer_mean=entity_value,
                peer_median=entity_value,
                peer_best=entity_value,
                peer_worst=entity_value,
                gap_to_best=0.0,
                gap_to_median=0.0,
                narrative="No peer data available for comparison."
            )

        # Get direction for ranking
        metric_data = BANKING_BENCHMARKS_2024.get(metric_id, {})
        good_direction = metric_data.get("good_direction", "higher")

        # Add entity to peer list for ranking
        all_values = [(entity_name, entity_value)] + peer_data
        reverse = good_direction == "higher"
        sorted_peers = sorted(all_values, key=lambda x: x[1], reverse=reverse)

        # Find entity's rank
        rank = next(i + 1 for i, (name, _) in enumerate(sorted_peers) if name == entity_name)
        total = len(sorted_peers)

        # Calculate percentile (higher is better regardless of direction)
        percentile = 100 * (total - rank) / (total - 1) if total > 1 else 50.0

        # Statistics
        peer_values_only = [v for _, v in peer_data]
        peer_mean = sum(peer_values_only) / len(peer_values_only)
        peer_median = sorted(peer_values_only)[len(peer_values_only) // 2]
        peer_best = sorted_peers[0][1]
        peer_worst = sorted_peers[-1][1]

        # Gap analysis
        gap_to_best = entity_value - peer_best if good_direction == "higher" else peer_best - entity_value
        gap_to_median = entity_value - peer_median if good_direction == "higher" else peer_median - entity_value

        # Generate narrative
        narrative = self._generate_ranking_narrative(
            entity_name, rank, total, metric_name,
            entity_value, peer_median, gap_to_best, good_direction
        )

        return PeerRanking(
            entity_id=entity_id,
            entity_name=entity_name,
            metric_id=metric_id,
            metric_name=metric_name,
            rank=rank,
            total_peers=total,
            percentile=round(percentile, 1),
            entity_value=entity_value,
            peer_values=peer_data[:10],  # Top 10 peers
            peer_mean=round(peer_mean, 2),
            peer_median=round(peer_median, 2),
            peer_best=peer_best,
            peer_worst=peer_worst,
            gap_to_best=round(gap_to_best, 2),
            gap_to_median=round(gap_to_median, 2),
            narrative=narrative
        )

    def _generate_ranking_narrative(
        self,
        entity_name: str,
        rank: int,
        total: int,
        metric_name: str,
        value: float,
        median: float,
        gap_to_best: float,
        direction: str
    ) -> str:
        """Generate ranking narrative."""
        percentile_position = 100 * (total - rank) / total

        if percentile_position >= 75:
            position_desc = "top quartile performer"
        elif percentile_position >= 50:
            position_desc = "above-median performer"
        elif percentile_position >= 25:
            position_desc = "below-median performer"
        else:
            position_desc = "bottom quartile performer"

        narrative = f"{entity_name} ranks #{rank} of {total} peers for {metric_name}, "
        narrative += f"positioning as a {position_desc}. "

        if abs(gap_to_best) > 0.01:
            if gap_to_best < 0:
                narrative += f"Gap to industry leader: {abs(gap_to_best):.2f}pp improvement needed. "
            else:
                narrative += f"Leading peers by {gap_to_best:.2f}pp. "

        return narrative

    def compare_all_metrics(
        self,
        entity_id: str,
        entity_name: str,
        metrics: Dict[str, float],
        institution_type: InstitutionType,
        period: str = "2024-Q4"
    ) -> PeerComparisonSummary:
        """
        Comprehensive peer comparison across all metrics.

        Args:
            entity_id: Entity identifier
            entity_name: Entity name
            metrics: Dict of metric_id -> value
            institution_type: Institution type
            period: Reporting period

        Returns:
            PeerComparisonSummary with full analysis
        """
        rankings: List[PeerRanking] = []

        for metric_id, value in metrics.items():
            metric_data = BANKING_BENCHMARKS_2024.get(metric_id, {})
            metric_name = metric_data.get("metric_name", metric_id)

            ranking = self.rank_against_peers(
                entity_id, entity_name, value,
                metric_id, metric_name,
                institution_type, period
            )
            rankings.append(ranking)

        if not rankings:
            return PeerComparisonSummary(
                entity_id=entity_id,
                entity_name=entity_name,
                peer_group=institution_type.value,
                total_peers=0,
                overall_rank=1,
                avg_percentile=50.0,
                metrics_in_top_quartile=0,
                metrics_in_bottom_quartile=0,
                rankings=[],
                narrative="No metrics available for comparison."
            )

        # Calculate overall statistics
        avg_percentile = sum(r.percentile for r in rankings) / len(rankings)
        avg_rank = sum(r.rank for r in rankings) / len(rankings)
        total_peers = rankings[0].total_peers if rankings else 0

        top_quartile = sum(1 for r in rankings if r.percentile >= 75)
        bottom_quartile = sum(1 for r in rankings if r.percentile < 25)

        # Identify strengths and weaknesses
        sorted_by_percentile = sorted(rankings, key=lambda r: r.percentile, reverse=True)
        strengths = [f"{r.metric_name} (#{r.rank})" for r in sorted_by_percentile[:3] if r.percentile >= 50]
        weaknesses = [f"{r.metric_name} (#{r.rank})" for r in sorted_by_percentile[-3:] if r.percentile < 50]

        # Key differentiators (metrics where rank is significantly different from average)
        differentiators = []
        for r in rankings:
            if r.percentile >= 80:
                differentiators.append(f"Strong in {r.metric_name} (Top 20%)")
            elif r.percentile <= 20:
                differentiators.append(f"Weak in {r.metric_name} (Bottom 20%)")

        # Generate narrative
        narrative = f"{entity_name} has an average percentile of {avg_percentile:.1f}% across {len(rankings)} metrics, "
        narrative += f"placing approximately #{int(avg_rank)} among {total_peers} peers. "

        if top_quartile > bottom_quartile:
            narrative += f"Overall competitive position is strong with {top_quartile} metrics in top quartile."
        elif bottom_quartile > top_quartile:
            narrative += f"Improvement needed in {bottom_quartile} metrics currently in bottom quartile."
        else:
            narrative += "Competitive position is balanced across peer group."

        return PeerComparisonSummary(
            entity_id=entity_id,
            entity_name=entity_name,
            peer_group=institution_type.value,
            total_peers=total_peers,
            overall_rank=int(avg_rank),
            avg_percentile=round(avg_percentile, 1),
            metrics_in_top_quartile=top_quartile,
            metrics_in_bottom_quartile=bottom_quartile,
            rankings=rankings,
            competitive_strengths=strengths,
            competitive_weaknesses=weaknesses,
            key_differentiators=differentiators[:5],
            narrative=narrative
        )


# Export
__all__ = [
    # Enums
    "InstitutionType",
    "MetricCategory",
    "BenchmarkStatus",
    # Models
    "IndustryBenchmark",
    "EntityMetric",
    "BenchmarkResult",
    "BenchmarkSummary",
    # Historical Models
    "HistoricalDataPoint",
    "HistoricalTrend",
    "PeerRanking",
    "PeerComparisonSummary",
    # Constants
    "BANKING_BENCHMARKS_2024",
    "INSURANCE_BENCHMARKS_2024",
    # Classes
    "IJKBenchmarkEngine",
    "OJKDataFetcher",
    "HistoricalTrendAnalyzer",
    "PeerRankingEngine",
    # Functions
    "generate_sample_entity_metrics",
]
