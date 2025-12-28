"""
Tests for new AURIX v4.2 modules:
- IJK Benchmarking
- Macro-Financial Stress Tester
- Risk Habit Scorecard
"""

import pytest
from datetime import datetime, date, timedelta

# ============================================
# IJK Benchmarking Tests
# ============================================

from modules.ijk_benchmarking import (
    InstitutionType,
    MetricCategory,
    BenchmarkStatus,
    IndustryBenchmark,
    EntityMetric,
    BenchmarkResult,
    BenchmarkSummary,
    IJKBenchmarkEngine,
    generate_sample_entity_metrics,
    BANKING_BENCHMARKS_2024,
)


class TestIJKBenchmarking:
    """Tests for IJK Benchmarking module."""

    @pytest.fixture
    def engine(self):
        """Create benchmark engine instance."""
        return IJKBenchmarkEngine()

    def test_institution_types(self):
        """Test institution type enum values."""
        assert InstitutionType.BANK_BUKU4 == "bank_buku4"
        assert InstitutionType.INSURANCE_LIFE == "insurance_life"

    def test_banking_benchmarks_data(self):
        """Test banking benchmark data exists."""
        assert "car" in BANKING_BENCHMARKS_2024
        assert "npl_gross" in BANKING_BENCHMARKS_2024
        assert "ldr" in BANKING_BENCHMARKS_2024

        car_data = BANKING_BENCHMARKS_2024["car"]
        assert car_data["regulatory_min"] == 8.0
        assert "buku4" in car_data

    def test_get_benchmark_data(self, engine):
        """Test getting benchmark data."""
        benchmark = engine.get_benchmark_data(
            "car",
            InstitutionType.BANK_BUKU4,
            "2024-Q4"
        )

        assert benchmark is not None
        assert benchmark.metric_id == "car"
        assert benchmark.regulatory_min == 8.0
        assert benchmark.industry_mean > 0

    def test_benchmark_metric(self, engine):
        """Test benchmarking a single metric."""
        entity_metric = EntityMetric(
            entity_id="BANK-001",
            entity_name="Test Bank",
            metric_id="car",
            value=22.0,
            period="2024-Q4"
        )

        result = engine.benchmark_metric(entity_metric, InstitutionType.BANK_BUKU4)

        assert result is not None
        assert result.entity_value == 22.0
        assert result.percentile_rank >= 0
        assert result.percentile_rank <= 100
        assert result.regulatory_compliant is True  # CAR 22% > 8% min

    def test_benchmark_below_regulatory(self, engine):
        """Test benchmarking a metric below regulatory threshold."""
        entity_metric = EntityMetric(
            entity_id="BANK-002",
            entity_name="Struggling Bank",
            metric_id="car",
            value=7.5,  # Below 8% minimum
            period="2024-Q4"
        )

        result = engine.benchmark_metric(entity_metric, InstitutionType.BANK_BUKU4)

        assert result is not None
        assert result.regulatory_compliant is False
        assert result.status == BenchmarkStatus.CONCERN

    def test_benchmark_entity_full(self, engine):
        """Test full entity benchmarking."""
        metrics = generate_sample_entity_metrics()

        summary = engine.benchmark_entity(
            entity_id="BANK-001",
            entity_name="Sample Bank",
            institution_type=InstitutionType.BANK_BUKU3,
            metrics=metrics
        )

        assert isinstance(summary, BenchmarkSummary)
        assert summary.overall_percentile >= 0
        assert len(summary.results) > 0


# ============================================
# Macro-Financial Stress Tester Tests
# ============================================

from modules.stress_tester import (
    ScenarioSeverity,
    RiskType,
    StressOutcome,
    PortfolioPosition,
    StressScenario,
    MacroStressTester,
    generate_sample_portfolio,
    STRESS_SCENARIOS,
    REGULATORY_THRESHOLDS,
)


class TestStressTester:
    """Tests for Macro-Financial Stress Tester."""

    @pytest.fixture
    def tester(self):
        """Create stress tester instance."""
        return MacroStressTester()

    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio."""
        return generate_sample_portfolio()

    def test_scenario_severity_enum(self):
        """Test scenario severity enum."""
        assert ScenarioSeverity.MILD == "mild"
        assert ScenarioSeverity.EXTREME == "extreme"

    def test_predefined_scenarios_exist(self):
        """Test predefined stress scenarios exist."""
        assert "bi_rate_mild" in STRESS_SCENARIOS
        assert "rupiah_severe" in STRESS_SCENARIOS
        assert "combined_severe" in STRESS_SCENARIOS

    def test_regulatory_thresholds(self):
        """Test regulatory threshold values."""
        assert REGULATORY_THRESHOLDS["car_minimum"] == 8.0
        assert REGULATORY_THRESHOLDS["npl_maximum"] == 5.0

    def test_portfolio_car_calculation(self, sample_portfolio):
        """Test portfolio CAR calculation."""
        car = sample_portfolio.current_car

        assert car > 0
        assert car < 100  # Reasonable range

    def test_run_stress_test_mild(self, tester, sample_portfolio):
        """Test running mild stress scenario."""
        scenario = STRESS_SCENARIOS["bi_rate_mild"]
        result = tester.run_stress_test(sample_portfolio, scenario)

        assert result is not None
        assert result.baseline_car > 0
        assert result.projections.projected_car > 0
        assert result.projections.car_change <= 0  # CAR should decrease under stress
        assert result.narrative != ""

    def test_run_stress_test_severe(self, tester, sample_portfolio):
        """Test running severe stress scenario."""
        scenario = STRESS_SCENARIOS["combined_severe"]
        result = tester.run_stress_test(sample_portfolio, scenario)

        assert result is not None
        assert len(result.impacts) > 0
        assert result.total_loss > 0
        # Severe scenario should have larger impact
        assert abs(result.projections.car_change) > abs(
            tester.run_stress_test(
                sample_portfolio,
                STRESS_SCENARIOS["bi_rate_mild"]
            ).projections.car_change
        )

    def test_stress_test_suite(self, tester, sample_portfolio):
        """Test running full stress test suite."""
        suite = tester.run_stress_test_suite(
            entity_name="Test Bank",
            position=sample_portfolio,
            scenario_ids=["bi_rate_mild", "bi_rate_moderate"]
        )

        assert suite.scenarios_tested == 2
        assert len(suite.results) == 2
        assert suite.worst_case_car > 0
        # pass_count + fail_count may not equal total as MARGINAL outcomes exist
        assert suite.pass_count >= 0
        assert suite.fail_count >= 0
        assert suite.overall_resilience != ""


# ============================================
# Risk Habit Scorecard Tests
# ============================================

from modules.risk_habit_scorecard import (
    HabitCategory,
    HabitFrequency,
    ScoreLevel,
    TrendDirection,
    HabitExecution,
    UserScorecard,
    RiskHabitEngine,
    generate_sample_executions,
    RISK_HABITS,
)


class TestRiskHabitScorecard:
    """Tests for Risk Habit Scorecard."""

    @pytest.fixture
    def engine(self):
        """Create habit engine instance."""
        return RiskHabitEngine()

    def test_habit_category_enum(self):
        """Test habit category enum."""
        assert HabitCategory.FINDING_MANAGEMENT == "finding_management"
        assert HabitCategory.RISK_REPORTING == "risk_reporting"

    def test_risk_habits_defined(self):
        """Test risk habits are defined."""
        assert "finding_closure" in RISK_HABITS
        assert "kri_monitoring" in RISK_HABITS

        finding_habit = RISK_HABITS["finding_closure"]
        assert "name" in finding_habit
        assert "weight" in finding_habit
        assert "atomic_cue" in finding_habit

    def test_generate_sample_executions(self):
        """Test sample execution generation."""
        executions = generate_sample_executions(
            user_id="USER-001",
            days=30,
            completion_rate=0.8
        )

        assert len(executions) > 0
        assert all(isinstance(e, HabitExecution) for e in executions)
        assert all(e.user_id == "USER-001" for e in executions)

    def test_calculate_habit_score(self, engine):
        """Test habit score calculation."""
        executions = [
            HabitExecution(
                execution_id=f"EXEC-{i}",
                habit_id="finding_closure",
                user_id="USER-001",
                execution_date=date.today() - timedelta(days=i * 7),
                completed=True,
                on_time=True
            )
            for i in range(4)
        ]

        compliance_rate, streak, weighted_score = engine.calculate_habit_score(
            "finding_closure",
            executions,
            period_days=30
        )

        assert compliance_rate >= 0
        assert compliance_rate <= 100
        assert streak >= 0
        assert weighted_score >= 0

    def test_generate_user_scorecard(self, engine):
        """Test user scorecard generation."""
        executions = generate_sample_executions(
            user_id="USER-001",
            days=30,
            completion_rate=0.85
        )

        scorecard = engine.generate_user_scorecard(
            user_id="USER-001",
            user_name="John Auditor",
            department="Internal Audit",
            executions=executions,
            period_start=date.today() - timedelta(days=30),
            period_end=date.today()
        )

        assert isinstance(scorecard, UserScorecard)
        assert scorecard.user_id == "USER-001"
        assert scorecard.overall_score >= 0
        assert scorecard.overall_score <= 100
        assert len(scorecard.habit_scores) > 0
        assert scorecard.score_level in ScoreLevel

    def test_score_level_determination(self, engine):
        """Test score level determination."""
        assert engine._get_score_level(95) == ScoreLevel.EXCELLENT
        assert engine._get_score_level(80) == ScoreLevel.GOOD
        assert engine._get_score_level(65) == ScoreLevel.ADEQUATE
        assert engine._get_score_level(45) == ScoreLevel.NEEDS_IMPROVEMENT
        assert engine._get_score_level(30) == ScoreLevel.POOR

    def test_compliance_nudge_generation(self, engine):
        """Test compliance nudge generation."""
        nudge = engine.generate_compliance_nudge(
            user_id="USER-001",
            habit_id="finding_closure",
            nudge_type="reminder"
        )

        assert nudge is not None
        assert nudge.user_id == "USER-001"
        assert nudge.habit_id == "finding_closure"
        assert len(nudge.message) > 0

    def test_team_scorecard(self, engine):
        """Test team scorecard generation."""
        # Generate scorecards for multiple users
        user_scorecards = []
        for i in range(5):
            executions = generate_sample_executions(
                user_id=f"USER-{i:03d}",
                days=30,
                completion_rate=0.7 + (i * 0.05)  # Varying completion rates
            )

            scorecard = engine.generate_user_scorecard(
                user_id=f"USER-{i:03d}",
                user_name=f"Auditor {i}",
                department="Internal Audit",
                executions=executions,
                period_start=date.today() - timedelta(days=30),
                period_end=date.today()
            )
            user_scorecards.append(scorecard)

        team_scorecard = engine.generate_team_scorecard(
            team_id="TEAM-001",
            team_name="Audit Team A",
            user_scorecards=user_scorecards
        )

        assert team_scorecard.team_size == 5
        assert team_scorecard.average_score >= 0
        assert team_scorecard.average_score <= 100
        assert len(team_scorecard.member_scorecards) == 5


# ============================================
# Integration Tests
# ============================================

class TestModuleIntegration:
    """Integration tests across new modules."""

    def test_modules_import_correctly(self):
        """Test all modules can be imported."""
        from modules import (
            # IJK Benchmarking
            IJKBenchmarkEngine,
            InstitutionType,
            # Stress Tester
            MacroStressTester,
            STRESS_SCENARIOS,
            # Risk Habit
            RiskHabitEngine,
            RISK_HABITS,
        )

        assert IJKBenchmarkEngine is not None
        assert MacroStressTester is not None
        assert RiskHabitEngine is not None

    def test_stress_test_with_benchmark(self):
        """Test stress test results can inform benchmark analysis."""
        from modules import (
            MacroStressTester,
            generate_sample_portfolio,
            IJKBenchmarkEngine,
            InstitutionType,
            STRESS_SCENARIOS,
        )

        # Run stress test
        tester = MacroStressTester()
        portfolio = generate_sample_portfolio()
        result = tester.run_stress_test(portfolio, STRESS_SCENARIOS["bi_rate_moderate"])

        # Use projected CAR in benchmark
        engine = IJKBenchmarkEngine()
        from modules.ijk_benchmarking import EntityMetric

        projected_car = result.projections.projected_car

        entity_metric = EntityMetric(
            entity_id="BANK-001",
            entity_name="Test Bank",
            metric_id="car",
            value=projected_car,
            period="2024-Q4"
        )

        benchmark = engine.benchmark_metric(entity_metric, InstitutionType.BANK_BUKU3)

        assert benchmark is not None
        # Stressed CAR should still be benchmarkable
        assert benchmark.percentile_rank >= 0
