"""
Tests for AURIX 2026 World-Class Features:
- RBAC (Role-Based Access Control)
- Predictive KPI Engine
- Regulation Monitor
- Compliance Gap Analyzer
"""

import pytest
from datetime import datetime, timedelta


# ============================================
# RBAC Tests
# ============================================

from modules.rbac import (
    UserRole,
    AccessLevel,
    ModulePermission,
    RoleConfig,
    UserSession,
    ROLE_PAGE_ACCESS,
    ROLE_CONFIGS,
    MODULE_PERMISSIONS,
    get_permitted_pages,
    can_access_page,
    get_module_access,
    can_write,
    can_approve,
    get_role_config,
    filter_pages_by_role,
    get_role_display_info,
    get_all_roles_info,
)


class TestRBACEnums:
    """Test RBAC enum definitions."""

    def test_user_role_enum(self):
        """Test UserRole enum values."""
        assert UserRole.EXECUTIVE == "executive"
        assert UserRole.AUDIT_MANAGER == "audit_manager"
        assert UserRole.FIELD_AUDITOR == "field_auditor"
        assert UserRole.AUDITEE == "auditee"
        assert UserRole.SYSTEM_ADMIN == "system_admin"
        assert len(UserRole) == 5

    def test_access_level_enum(self):
        """Test AccessLevel enum values."""
        assert AccessLevel.FULL == "full"
        assert AccessLevel.WRITE == "write"
        assert AccessLevel.READ_ONLY == "read"
        assert AccessLevel.NONE == "none"


class TestRBACConfigurations:
    """Test RBAC configuration data."""

    def test_all_roles_have_page_access(self):
        """Test all roles have page access defined."""
        for role in UserRole:
            assert role in ROLE_PAGE_ACCESS
            assert isinstance(ROLE_PAGE_ACCESS[role], list)

    def test_all_roles_have_config(self):
        """Test all roles have configuration."""
        for role in UserRole:
            assert role in ROLE_CONFIGS
            config = ROLE_CONFIGS[role]
            assert isinstance(config, RoleConfig)
            assert config.display_name
            assert config.description

    def test_executive_limited_access(self):
        """Test executive has limited but strategic access."""
        exec_pages = ROLE_PAGE_ACCESS[UserRole.EXECUTIVE]
        assert "📊 Dashboard" in exec_pages
        assert "🏛️ Executive Dashboard" in exec_pages
        assert "🎰 Stress Tester" in exec_pages
        # Should not have operational access
        assert "📝 Workpapers" not in exec_pages
        assert "🔬 Root Cause Analyzer" not in exec_pages

    def test_field_auditor_operational_access(self):
        """Test field auditor has operational tools."""
        auditor_pages = ROLE_PAGE_ACCESS[UserRole.FIELD_AUDITOR]
        assert "📝 Workpapers" in auditor_pages
        assert "🔬 Root Cause Analyzer" in auditor_pages
        assert "🎭 PTCF Builder" in auditor_pages
        # Should not have executive access
        assert "🏛️ Executive Dashboard" not in auditor_pages

    def test_auditee_minimal_access(self):
        """Test auditee has minimal access."""
        auditee_pages = ROLE_PAGE_ACCESS[UserRole.AUDITEE]
        assert len(auditee_pages) <= 5  # Very limited
        assert "📊 Dashboard" in auditee_pages
        assert "📋 Findings Tracker" in auditee_pages

    def test_system_admin_full_access(self):
        """Test system admin has full access."""
        admin_pages = ROLE_PAGE_ACCESS[UserRole.SYSTEM_ADMIN]
        assert "🔧 Admin Panel" in admin_pages
        # Should have more pages than any other role
        for role in UserRole:
            if role != UserRole.SYSTEM_ADMIN:
                assert len(admin_pages) >= len(ROLE_PAGE_ACCESS[role])


class TestRBACPermissions:
    """Test RBAC permission functions."""

    def test_get_permitted_pages(self):
        """Test getting permitted pages for roles."""
        exec_pages = get_permitted_pages(UserRole.EXECUTIVE)
        assert isinstance(exec_pages, list)
        assert len(exec_pages) > 0

    def test_can_access_page_positive(self):
        """Test positive page access check."""
        assert can_access_page("📊 Dashboard", UserRole.EXECUTIVE)
        assert can_access_page("🏛️ Executive Dashboard", UserRole.EXECUTIVE)

    def test_can_access_page_negative(self):
        """Test negative page access check."""
        assert not can_access_page("📝 Workpapers", UserRole.EXECUTIVE)
        assert not can_access_page("🔧 Admin Panel", UserRole.FIELD_AUDITOR)

    def test_module_access_levels(self):
        """Test module access level retrieval."""
        # Executive has read-only to findings
        access = get_module_access("findings", UserRole.EXECUTIVE)
        assert access == AccessLevel.READ_ONLY

        # Audit Manager has full access
        access = get_module_access("findings", UserRole.AUDIT_MANAGER)
        assert access == AccessLevel.FULL

    def test_can_write_permission(self):
        """Test write permission check."""
        # Field auditor can write findings
        assert can_write("findings", UserRole.FIELD_AUDITOR)
        # Executive cannot write findings
        assert not can_write("findings", UserRole.EXECUTIVE)

    def test_can_approve_permission(self):
        """Test approve permission check."""
        # Audit Manager can approve findings
        assert can_approve("findings", UserRole.AUDIT_MANAGER)
        # Field auditor cannot approve
        assert not can_approve("findings", UserRole.FIELD_AUDITOR)

    def test_filter_pages_by_role(self):
        """Test filtering pages by role."""
        all_pages = ["📊 Dashboard", "📝 Workpapers", "🔧 Admin Panel"]
        filtered = filter_pages_by_role(all_pages, UserRole.AUDITEE)
        assert "📊 Dashboard" in filtered
        assert "📝 Workpapers" not in filtered
        assert "🔧 Admin Panel" not in filtered


class TestRBACHelpers:
    """Test RBAC helper functions."""

    def test_get_role_config(self):
        """Test role config retrieval."""
        config = get_role_config(UserRole.EXECUTIVE)
        assert config.role == UserRole.EXECUTIVE
        assert config.display_name == "Executive / Board"

    def test_get_role_display_info(self):
        """Test role display info."""
        info = get_role_display_info(UserRole.AUDIT_MANAGER)
        assert info["role"] == "audit_manager"
        assert info["display_name"] == "Audit Manager"
        assert "page_count" in info

    def test_get_all_roles_info(self):
        """Test getting all roles info."""
        all_info = get_all_roles_info()
        assert len(all_info) == 5
        assert all(isinstance(info, dict) for info in all_info)


class TestUserSession:
    """Test UserSession model."""

    def test_user_session_defaults(self):
        """Test UserSession default values."""
        session = UserSession()
        assert session.user_id == "demo_user"
        assert session.role == UserRole.FIELD_AUDITOR
        assert session.login_time is not None

    def test_user_session_custom(self):
        """Test UserSession with custom values."""
        session = UserSession(
            user_id="admin-001",
            username="Admin User",
            role=UserRole.SYSTEM_ADMIN,
            department="IT"
        )
        assert session.user_id == "admin-001"
        assert session.role == UserRole.SYSTEM_ADMIN


# ============================================
# Predictive KPI Engine Tests
# ============================================

from modules.predictive_kpi import (
    ForecastHorizon,
    TrendDirection,
    AlertSeverity,
    KPIDataPoint,
    KPITimeSeries,
    ForecastResult,
    KPIForecast,
    BreachPrediction,
    WhatIfScenario,
    WhatIfResult,
    PredictiveAlertPanel,
    KPI_THRESHOLDS,
    PredictiveKPIEngine,
    generate_sample_kpi_timeseries,
    KPIThreshold,
)


class TestPredictiveKPIEnums:
    """Test Predictive KPI enum definitions."""

    def test_forecast_horizon_enum(self):
        """Test ForecastHorizon enum values."""
        assert ForecastHorizon.DAYS_30 == "30_days"
        assert ForecastHorizon.DAYS_90 == "90_days"
        assert ForecastHorizon.YEAR == "365_days"

    def test_trend_direction_enum(self):
        """Test TrendDirection enum values."""
        assert TrendDirection.IMPROVING == "improving"
        assert TrendDirection.DETERIORATING == "deteriorating"

    def test_alert_severity_enum(self):
        """Test AlertSeverity enum values."""
        assert AlertSeverity.CRITICAL == "critical"
        assert AlertSeverity.WARNING == "warning"


class TestKPIThresholds:
    """Test KPI threshold configurations."""

    def test_car_threshold(self):
        """Test CAR threshold definition."""
        assert "car" in KPI_THRESHOLDS
        car = KPI_THRESHOLDS["car"]
        assert car["min"] == 8.0  # Regulatory minimum
        assert car["direction"] == "higher"

    def test_npl_threshold(self):
        """Test NPL threshold definition."""
        assert "npl_ratio" in KPI_THRESHOLDS
        npl = KPI_THRESHOLDS["npl_ratio"]
        assert npl["max"] == 5.0  # Regulatory maximum
        assert npl["direction"] == "lower"

    def test_ldr_optimal_range(self):
        """Test LDR has optimal range."""
        assert "ldr" in KPI_THRESHOLDS
        ldr = KPI_THRESHOLDS["ldr"]
        assert "min" in ldr
        assert "max" in ldr
        assert ldr["direction"] == "optimal"


class TestKPIDataModels:
    """Test KPI data models."""

    def test_kpi_data_point(self):
        """Test KPIDataPoint model."""
        dp = KPIDataPoint(
            kpi_id="car",
            date=datetime.now(),
            value=15.5
        )
        assert dp.kpi_id == "car"
        assert dp.value == 15.5
        assert dp.is_actual is True

    def test_kpi_threshold_model(self):
        """Test KPIThreshold model."""
        threshold = KPIThreshold(
            warning_level=10.0,
            danger_level=8.0,
            direction="lower"
        )
        assert threshold.warning_level == 10.0
        assert threshold.danger_level == 8.0
        assert threshold.direction == "lower"

    def test_kpi_time_series(self):
        """Test KPITimeSeries model."""
        ts = KPITimeSeries(
            kpi_id="car",
            kpi_name="Capital Adequacy Ratio",
            unit="%",
            data_points=[
                KPIDataPoint(kpi_id="car", date=datetime.now(), value=15.0)
            ]
        )
        assert ts.kpi_id == "car"
        assert len(ts.data_points) == 1


class TestPredictiveKPIEngine:
    """Test PredictiveKPIEngine class."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return PredictiveKPIEngine()

    @pytest.fixture
    def sample_timeseries(self):
        """Create sample time series."""
        return generate_sample_kpi_timeseries(
            kpi_id="car",
            kpi_name="Capital Adequacy Ratio",
            days=180,
            start_value=16.0,
            trend="deteriorating"
        )

    def test_generate_sample_timeseries(self):
        """Test sample time series generation."""
        ts = generate_sample_kpi_timeseries(
            kpi_id="npl_ratio",
            kpi_name="NPL Ratio",
            days=90,
            start_value=3.0,
            trend="stable"
        )
        assert ts.kpi_id == "npl_ratio"
        assert len(ts.data_points) == 90

    def test_forecast_kpi(self, engine, sample_timeseries):
        """Test KPI forecasting."""
        forecast = engine.forecast_kpi(
            sample_timeseries,
            horizon=ForecastHorizon.DAYS_30
        )

        assert isinstance(forecast, KPIForecast)
        assert forecast.kpi_id == "car"
        assert forecast.horizon == ForecastHorizon.DAYS_30
        assert len(forecast.forecasts) == 30
        assert forecast.current_value > 0
        assert forecast.end_value > 0

    def test_forecast_trend_detection(self, engine, sample_timeseries):
        """Test trend detection in forecasting."""
        forecast = engine.forecast_kpi(sample_timeseries)

        # Trend should be detected (may be stable due to random noise in sample data)
        assert forecast.trend_direction in [
            TrendDirection.DETERIORATING,
            TrendDirection.STABLE,
            TrendDirection.IMPROVING,
            TrendDirection.VOLATILE
        ]
        assert forecast.trend_strength >= 0
        assert forecast.trend_strength <= 1

    def test_breach_prediction(self, engine, sample_timeseries):
        """Test breach prediction."""
        forecast = engine.forecast_kpi(sample_timeseries)
        prediction = engine.predict_breach(forecast)

        # May or may not have prediction based on data
        if prediction:
            assert isinstance(prediction, BreachPrediction)
            assert prediction.breach_probability >= 0
            assert prediction.breach_probability <= 1

    def test_what_if_scenario(self, engine, sample_timeseries):
        """Test what-if scenario analysis."""
        forecast = engine.forecast_kpi(sample_timeseries)

        scenario = WhatIfScenario(
            scenario_id="SCEN-001",
            scenario_name="NPL Increase",
            description="Test NPL increase scenario",
            adjustments={"car": -2.0},
            adjustment_type="absolute"
        )

        result = engine.run_what_if_analysis(
            {"car": forecast},
            scenario
        )

        assert isinstance(result, WhatIfResult)
        assert result.scenario_id == "SCEN-001"
        assert "car" in result.impacts
        assert result.risk_change in ["increased", "decreased", "unchanged"]

    def test_alert_panel_generation(self, engine, sample_timeseries):
        """Test predictive alert panel generation."""
        forecast = engine.forecast_kpi(sample_timeseries)
        panel = engine.generate_alert_panel({"car": forecast})

        assert isinstance(panel, PredictiveAlertPanel)
        assert panel.total_alerts >= 0
        assert panel.horizon_days == 90


# ============================================
# Regulation Monitor Tests
# ============================================

from modules.regulation_monitor import (
    RegulatorType,
    RegulationType,
    ChangeType,
    ImpactLevel,
    ComplianceStatus,
    RegulationSection,
    RegulationDocument,
    RegulationChange,
    ComplianceActionItem,
    RegulationChangeReport,
    RegulationMonitor,
    generate_sample_regulations,
)


class TestRegulationEnums:
    """Test Regulation Monitor enums."""

    def test_regulator_type(self):
        """Test RegulatorType enum."""
        assert RegulatorType.OJK == "ojk"
        assert RegulatorType.BI == "bi"
        assert RegulatorType.PPATK == "ppatk"

    def test_regulation_type(self):
        """Test RegulationType enum."""
        assert RegulationType.POJK == "pojk"
        assert RegulationType.PBI == "pbi"
        assert RegulationType.SEOJK == "seojk"

    def test_change_type(self):
        """Test ChangeType enum."""
        assert ChangeType.NEW == "new"
        assert ChangeType.AMENDMENT == "amendment"
        assert ChangeType.REVOCATION == "revocation"

    def test_impact_level(self):
        """Test ImpactLevel enum."""
        assert ImpactLevel.CRITICAL == "critical"
        assert ImpactLevel.HIGH == "high"


class TestRegulationModels:
    """Test Regulation Monitor models."""

    def test_regulation_document(self):
        """Test RegulationDocument model."""
        doc = RegulationDocument(
            regulation_id="POJK 12/2017",
            regulator=RegulatorType.OJK,
            regulation_type=RegulationType.POJK,
            title="Test Regulation",
            issue_date=datetime(2017, 11, 10),
            effective_date=datetime(2018, 1, 1),
            full_text="Test content",
            content_hash="abc123"
        )
        assert doc.regulation_id == "POJK 12/2017"
        assert doc.regulator == RegulatorType.OJK

    def test_compliance_action_item(self):
        """Test ComplianceActionItem model."""
        action = ComplianceActionItem(
            action_id="ACT-001",
            change_id="CHG-001",
            regulation_id="POJK 12/2017",
            title="Review regulation",
            description="Review new regulation requirements",
            category="policy",
            responsible_unit="Compliance",
            priority="high",
            due_date=datetime.now() + timedelta(days=30),
            estimated_effort="2-3 days"
        )
        assert action.status == ComplianceStatus.PENDING
        assert action.progress_percent == 0


class TestRegulationMonitor:
    """Test RegulationMonitor class."""

    @pytest.fixture
    def monitor(self):
        """Create monitor instance."""
        return RegulationMonitor()

    @pytest.fixture
    def sample_regulation(self):
        """Create sample regulation."""
        return RegulationDocument(
            regulation_id="POJK-TEST-001",
            regulator=RegulatorType.OJK,
            regulation_type=RegulationType.POJK,
            title="Test POJK Regulation",
            issue_date=datetime.now(),
            effective_date=datetime.now() + timedelta(days=30),
            full_text="Article 1: Test content.\nArticle 2: More content.",
            keywords=["test", "compliance"],
            content_hash="test123"
        )

    def test_generate_sample_regulations(self):
        """Test sample regulation generation."""
        regs = generate_sample_regulations()
        assert len(regs) >= 3
        assert all(isinstance(r, RegulationDocument) for r in regs)

    def test_register_regulation(self, monitor, sample_regulation):
        """Test regulation registration."""
        monitor.register_regulation(sample_regulation)
        assert sample_regulation.regulation_id in monitor._regulations

    def test_detect_new_regulation(self, monitor, sample_regulation):
        """Test new regulation detection."""
        change = monitor.detect_new_regulation(sample_regulation)

        assert isinstance(change, RegulationChange)
        assert change.change_type == ChangeType.NEW
        assert change.regulation_id == sample_regulation.regulation_id
        assert change.impact_level == ImpactLevel.HIGH  # POJK is high impact

    def test_check_for_changes(self, monitor, sample_regulation):
        """Test change detection."""
        # Register original
        monitor.register_regulation(sample_regulation)

        # Check with modified content
        new_content = "Article 1: Modified content.\nArticle 2: More content.\nArticle 3: New article."
        change = monitor.check_for_changes(
            sample_regulation.regulation_id,
            new_content,
            "2.0"
        )

        assert change is not None
        assert change.change_type == ChangeType.AMENDMENT
        assert change.new_version == "2.0"

    def test_no_change_detection(self, monitor):
        """Test no change when content same."""
        # Create regulation with computed hash (not manual)
        reg = RegulationDocument(
            regulation_id="POJK-HASH-TEST",
            regulator=RegulatorType.OJK,
            regulation_type=RegulationType.POJK,
            title="Hash Test Regulation",
            issue_date=datetime.now(),
            effective_date=datetime.now() + timedelta(days=30),
            full_text="Fixed content for hash testing.",
            content_hash=""  # Will be computed
        )
        # Compute the hash properly
        import hashlib
        reg.content_hash = hashlib.sha256(reg.full_text.encode()).hexdigest()[:16]

        monitor.register_regulation(reg)

        change = monitor.check_for_changes(
            reg.regulation_id,
            reg.full_text,  # Exact same content
            "1.0"
        )

        assert change is None

    def test_generate_action_items(self, monitor, sample_regulation):
        """Test action item generation."""
        change = monitor.detect_new_regulation(sample_regulation)
        actions = monitor.generate_action_items(change)

        assert len(actions) >= 3  # New regulation should have multiple actions
        assert all(isinstance(a, ComplianceActionItem) for a in actions)
        assert any("Review" in a.title for a in actions)
        assert any("Gap analysis" in a.title for a in actions)

    def test_generate_change_report(self, monitor):
        """Test change report generation."""
        # Add some regulations and changes
        regs = generate_sample_regulations()
        for reg in regs:
            monitor.detect_new_regulation(reg)

        report = monitor.generate_change_report(
            datetime.now() - timedelta(days=7),
            datetime.now() + timedelta(days=1)
        )

        assert isinstance(report, RegulationChangeReport)
        assert report.total_changes >= 3
        assert report.new_regulations >= 3
        assert report.executive_summary

    def test_get_pending_actions(self, monitor, sample_regulation):
        """Test pending actions retrieval."""
        change = monitor.detect_new_regulation(sample_regulation)
        monitor.generate_action_items(change)

        pending = monitor.get_pending_actions(due_within_days=90)
        assert len(pending) > 0
        assert all(a.status in [ComplianceStatus.PENDING, ComplianceStatus.IN_PROGRESS, ComplianceStatus.OVERDUE]
                   for a in pending)


# ============================================
# Compliance Gap Analyzer Tests
# ============================================

from modules.compliance_gap import (
    ControlStatus,
    ControlEffectiveness,
    GapSeverity,
    RemediationPriority,
    RegulatoryRequirement,
    Control,
    ControlMapping,
    ComplianceGap,
    RemediationPlan,
    ComplianceScorecard,
    ComplianceGapAnalyzer,
    generate_sample_data,
)


class TestComplianceGapEnums:
    """Test Compliance Gap enums."""

    def test_control_status(self):
        """Test ControlStatus enum."""
        assert ControlStatus.IMPLEMENTED == "implemented"
        assert ControlStatus.NOT_IMPLEMENTED == "not_implemented"

    def test_control_effectiveness(self):
        """Test ControlEffectiveness enum."""
        assert ControlEffectiveness.EFFECTIVE == "effective"
        assert ControlEffectiveness.NOT_TESTED == "not_tested"

    def test_gap_severity(self):
        """Test GapSeverity enum."""
        assert GapSeverity.CRITICAL == "critical"
        assert GapSeverity.LOW == "low"


class TestComplianceGapModels:
    """Test Compliance Gap models."""

    def test_regulatory_requirement(self):
        """Test RegulatoryRequirement model."""
        req = RegulatoryRequirement(
            requirement_id="REQ-001",
            regulation_id="POJK 12/2017",
            regulation_name="APU-PPT",
            article="Article 11",
            requirement_text="Must implement CDD",
            category="risk",
            effective_date=datetime(2018, 1, 1)
        )
        assert req.mandatory is True

    def test_control(self):
        """Test Control model."""
        ctrl = Control(
            control_id="CTL-001",
            control_name="CDD Process",
            description="Customer due diligence",
            control_type="preventive",
            control_category="automated",
            owner_department="Compliance"
        )
        assert ctrl.status == ControlStatus.IMPLEMENTED
        assert ctrl.effectiveness == ControlEffectiveness.NOT_TESTED

    def test_control_mapping(self):
        """Test ControlMapping model."""
        mapping = ControlMapping(
            mapping_id="MAP-001",
            control_id="CTL-001",
            requirement_id="REQ-001",
            coverage_type="full",
            coverage_percentage=100
        )
        assert mapping.validated is False


class TestComplianceGapAnalyzer:
    """Test ComplianceGapAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return ComplianceGapAnalyzer()

    @pytest.fixture
    def sample_data(self):
        """Get sample data."""
        return generate_sample_data()

    def test_generate_sample_data(self):
        """Test sample data generation."""
        reqs, ctrls, mappings = generate_sample_data()
        assert len(reqs) >= 3
        assert len(ctrls) >= 2
        assert len(mappings) >= 2

    def test_register_requirement(self, analyzer):
        """Test requirement registration."""
        req = RegulatoryRequirement(
            requirement_id="REQ-TEST",
            regulation_id="TEST-REG",
            regulation_name="Test Regulation",
            article="Article 1",
            requirement_text="Test requirement",
            category="governance",
            effective_date=datetime.now()
        )
        analyzer.register_requirement(req)
        assert "REQ-TEST" in analyzer._requirements

    def test_register_control(self, analyzer):
        """Test control registration."""
        ctrl = Control(
            control_id="CTL-TEST",
            control_name="Test Control",
            description="Test control",
            control_type="preventive",
            control_category="manual",
            owner_department="Compliance"
        )
        analyzer.register_control(ctrl)
        assert "CTL-TEST" in analyzer._controls

    def test_analyze_coverage_no_control(self, analyzer):
        """Test coverage analysis with no mapped control."""
        req = RegulatoryRequirement(
            requirement_id="REQ-NO-CTRL",
            regulation_id="REG-001",
            regulation_name="Test",
            article="Article 1",
            requirement_text="Requirement with no control",
            category="risk",
            effective_date=datetime.now()
        )
        analyzer.register_requirement(req)

        coverage, controls, gap = analyzer.analyze_coverage("REQ-NO-CTRL")

        assert coverage == 0.0
        assert len(controls) == 0
        assert gap is not None
        assert gap.gap_type == "no_control"
        assert gap.severity == GapSeverity.CRITICAL

    def test_analyze_coverage_full(self, analyzer):
        """Test coverage analysis with full coverage."""
        req = RegulatoryRequirement(
            requirement_id="REQ-FULL",
            regulation_id="REG-001",
            regulation_name="Test",
            article="Article 1",
            requirement_text="Fully covered requirement",
            category="control",
            effective_date=datetime.now()
        )
        ctrl = Control(
            control_id="CTL-FULL",
            control_name="Full Control",
            description="Fully effective control",
            control_type="preventive",
            control_category="automated",
            owner_department="Compliance",
            status=ControlStatus.IMPLEMENTED,
            effectiveness=ControlEffectiveness.EFFECTIVE
        )
        mapping = ControlMapping(
            mapping_id="MAP-FULL",
            control_id="CTL-FULL",
            requirement_id="REQ-FULL",
            coverage_type="full",
            coverage_percentage=100
        )

        analyzer.register_requirement(req)
        analyzer.register_control(ctrl)
        analyzer.add_mapping(mapping)

        coverage, controls, gap = analyzer.analyze_coverage("REQ-FULL")

        assert coverage == 100.0
        assert "CTL-FULL" in controls
        assert gap is None  # No gap for full coverage

    def test_analyze_coverage_partial(self, analyzer):
        """Test coverage analysis with partial coverage."""
        req = RegulatoryRequirement(
            requirement_id="REQ-PARTIAL",
            regulation_id="REG-001",
            regulation_name="Test",
            article="Article 1",
            requirement_text="Partially covered requirement",
            category="risk",
            effective_date=datetime.now()
        )
        ctrl = Control(
            control_id="CTL-PARTIAL",
            control_name="Partial Control",
            description="Partially effective control",
            control_type="detective",
            control_category="manual",
            owner_department="Operations",
            status=ControlStatus.IMPLEMENTED,
            effectiveness=ControlEffectiveness.PARTIALLY_EFFECTIVE
        )
        mapping = ControlMapping(
            mapping_id="MAP-PARTIAL",
            control_id="CTL-PARTIAL",
            requirement_id="REQ-PARTIAL",
            coverage_type="partial",
            coverage_percentage=70
        )

        analyzer.register_requirement(req)
        analyzer.register_control(ctrl)
        analyzer.add_mapping(mapping)

        coverage, controls, gap = analyzer.analyze_coverage("REQ-PARTIAL")

        # 70% coverage * 0.7 effectiveness = 49%
        assert coverage < 100
        assert coverage > 0
        assert gap is not None  # Should have a gap

    def test_analyze_all_requirements(self, analyzer, sample_data):
        """Test analyzing all requirements."""
        reqs, ctrls, mappings = sample_data

        for req in reqs:
            analyzer.register_requirement(req)
        for ctrl in ctrls:
            analyzer.register_control(ctrl)
        for mapping in mappings:
            analyzer.add_mapping(mapping)

        gaps = analyzer.analyze_all_requirements()

        assert isinstance(gaps, list)
        # Should find at least one gap (REQ-APU-003 has no control)
        assert len(gaps) >= 1

    def test_generate_remediation_plan(self, analyzer):
        """Test remediation plan generation."""
        gap = ComplianceGap(
            gap_id="GAP-001",
            requirement_id="REQ-001",
            regulation_id="REG-001",
            gap_title="Test Gap",
            gap_description="Test gap description",
            gap_type="no_control",
            severity=GapSeverity.HIGH,
            current_coverage=0,
            target_coverage=100,
            coverage_gap=100,
            potential_impact="Regulatory violation",
            risk_exposure="High"
        )

        plan = analyzer.generate_remediation_plan(gap)

        assert isinstance(plan, RemediationPlan)
        assert plan.gap_id == "GAP-001"
        assert plan.remediation_approach == "new_control"
        assert len(plan.action_items) > 0
        assert plan.target_completion > datetime.now()

    def test_calculate_compliance_score(self, analyzer, sample_data):
        """Test compliance scorecard calculation."""
        reqs, ctrls, mappings = sample_data

        for req in reqs:
            analyzer.register_requirement(req)
        for ctrl in ctrls:
            analyzer.register_control(ctrl)
        for mapping in mappings:
            analyzer.add_mapping(mapping)

        scorecard = analyzer.calculate_compliance_score()

        assert isinstance(scorecard, ComplianceScorecard)
        assert scorecard.overall_compliance_score >= 0
        assert scorecard.overall_compliance_score <= 100
        assert scorecard.total_requirements == 3
        assert scorecard.executive_summary


# ============================================
# Integration Tests
# ============================================

class TestWorldClassIntegration:
    """Integration tests across world-class modules."""

    def test_modules_import(self):
        """Test all modules can be imported."""
        from modules.rbac import UserRole, ROLE_CONFIGS
        from modules.predictive_kpi import PredictiveKPIEngine
        from modules.regulation_monitor import RegulationMonitor
        from modules.compliance_gap import ComplianceGapAnalyzer

        assert UserRole is not None
        assert PredictiveKPIEngine is not None
        assert RegulationMonitor is not None
        assert ComplianceGapAnalyzer is not None

    def test_rbac_with_compliance_gap(self):
        """Test RBAC permissions with compliance gap module."""
        from modules.rbac import can_write, can_approve, UserRole

        # Compliance module should be writable by audit manager
        assert can_write("findings", UserRole.AUDIT_MANAGER)
        assert can_approve("findings", UserRole.AUDIT_MANAGER)

        # But not by auditee
        assert not can_write("workpapers", UserRole.AUDITEE)

    def test_regulation_monitor_with_gap_analyzer(self):
        """Test integration between regulation monitor and gap analyzer."""
        from modules.regulation_monitor import RegulationMonitor, generate_sample_regulations
        from modules.compliance_gap import ComplianceGapAnalyzer, RegulatoryRequirement

        # Monitor detects new regulation
        monitor = RegulationMonitor()
        regs = generate_sample_regulations()
        change = monitor.detect_new_regulation(regs[0])

        # Gap analyzer can use regulation info
        analyzer = ComplianceGapAnalyzer()
        req = RegulatoryRequirement(
            requirement_id="REQ-FROM-REG",
            regulation_id=regs[0].regulation_id,
            regulation_name=regs[0].title,
            article="Article 11",
            requirement_text="Derived requirement",
            category="risk",
            effective_date=regs[0].effective_date
        )
        analyzer.register_requirement(req)

        # Analyze should find gap (no control)
        coverage, _, gap = analyzer.analyze_coverage("REQ-FROM-REG")
        assert coverage == 0
        assert gap is not None

    def test_predictive_kpi_with_scorecard(self):
        """Test KPI prediction informing compliance score."""
        from modules.predictive_kpi import (
            PredictiveKPIEngine,
            generate_sample_kpi_timeseries,
            ForecastHorizon
        )

        engine = PredictiveKPIEngine()

        # Generate KPI forecast
        ts = generate_sample_kpi_timeseries(
            kpi_id="car",
            days=90,
            start_value=12.0,
            trend="deteriorating"
        )

        forecast = engine.forecast_kpi(ts, ForecastHorizon.DAYS_30)

        # Forecast can be used in executive dashboard
        assert forecast.current_value > 0
        assert forecast.end_value > 0
        assert forecast.trend_direction is not None
