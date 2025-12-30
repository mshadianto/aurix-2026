"""
Executive Dashboard for AURIX Excellence 2026.

Premium executive-grade dashboard featuring:
- 5-Second Rule KPI Scorecard
- So-What Auto-Narrative insights
- Predictive KPI Alerts (AI-powered breach prediction)
- Flight Simulator scenario analysis
- Peer Industry Comparison
- Data Lineage transparency
- Export-ready reports

Designed for C-Suite and Senior Management.
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import random

from ui.styles.css_builder import get_current_theme
from ui.components import render_footer
from ui.components.executive import (
    KPIMetric,
    TrendDirection,
    DataLineage,
    render_executive_scorecard,
    render_so_what_panel,
    generate_so_what_insight,
    render_flight_simulator,
    render_data_lineage,
    render_export_panel,
)
from services.visitor_service import track_page_view
from modules.predictive_kpi import (
    PredictiveKPIEngine,
    KPITimeSeries,
    KPIThreshold,
    WhatIfScenario,
)


def render():
    """Render the Executive Dashboard."""
    t = get_current_theme()

    # Track page view
    track_page_view("Executive Dashboard")

    # Define executive KPIs
    kpis = _get_executive_kpis()

    # Render Executive Scorecard (5-Second Rule)
    render_executive_scorecard(
        metrics=kpis,
        title="Executive Command Center",
        subtitle="Real-time Risk & Performance Intelligence | AURIX Excellence 2026"
    )

    # Data Lineage for transparency
    lineage = DataLineage(
        source_system="Core Banking System (T24)",
        extraction_time=datetime.now() - timedelta(minutes=15),
        transformation="ETL Pipeline v3.2 → Data Warehouse → AURIX Analytics",
        quality_score=0.97,
        owner="Risk Management Division",
        refresh_frequency="Every 15 minutes"
    )
    render_data_lineage(lineage, compact=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main content in tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Strategic Insights",
        "🔮 Predictive Alerts",
        "🎮 Scenario Simulator",
        "📈 Detailed Analytics",
        "📑 Reports"
    ])

    with tab1:
        _render_strategic_insights(kpis, t)

    with tab2:
        _render_predictive_alerts(kpis, t)

    with tab3:
        _render_scenario_simulator(kpis, t)

    with tab4:
        _render_detailed_analytics(kpis, t)

    with tab5:
        _render_reports_section(kpis, t)

    # Footer
    render_footer()


def _get_executive_kpis() -> List[KPIMetric]:
    """Get executive KPI metrics with sample data."""
    return [
        KPIMetric(
            id="npl_ratio",
            label="NPL Ratio",
            value=5.54,
            unit="%",
            target=3.50,
            threshold_warning=4.00,
            threshold_danger=5.00,
            trend_direction=TrendDirection.UP,
            trend_value=0.82,
            period="vs Last Month",
            source="Core Banking - Credit Module",
            lower_is_better=True
        ),
        KPIMetric(
            id="ldr_ratio",
            label="LDR",
            value=94.32,
            unit="%",
            target=85.00,
            threshold_warning=90.00,
            threshold_danger=100.00,
            trend_direction=TrendDirection.UP,
            trend_value=2.15,
            period="vs Last Month",
            source="Treasury Management System",
            lower_is_better=True
        ),
        KPIMetric(
            id="car_ratio",
            label="CAR",
            value=18.45,
            unit="%",
            target=14.00,
            threshold_warning=12.00,
            threshold_danger=10.00,
            trend_direction=TrendDirection.DOWN,
            trend_value=0.35,
            period="vs Last Month",
            source="Risk Management System",
            lower_is_better=False
        ),
        KPIMetric(
            id="roa",
            label="ROA",
            value=2.18,
            unit="%",
            target=2.50,
            threshold_warning=1.50,
            threshold_danger=1.00,
            trend_direction=TrendDirection.STABLE,
            trend_value=0.05,
            period="vs Last Month",
            source="Finance & Accounting System",
            lower_is_better=False
        ),
        KPIMetric(
            id="lcr",
            label="LCR",
            value=142.50,
            unit="%",
            target=120.00,
            threshold_warning=110.00,
            threshold_danger=100.00,
            trend_direction=TrendDirection.UP,
            trend_value=5.20,
            period="vs Last Month",
            source="Treasury Management System",
            lower_is_better=False
        ),
        KPIMetric(
            id="cost_income",
            label="Cost-to-Income",
            value=48.75,
            unit="%",
            target=45.00,
            threshold_warning=50.00,
            threshold_danger=55.00,
            trend_direction=TrendDirection.DOWN,
            trend_value=1.25,
            period="vs Last Month",
            source="Finance & Accounting System",
            lower_is_better=True
        ),
    ]


def _render_strategic_insights(kpis: List[KPIMetric], t: dict):
    """Render strategic insights with So-What narratives."""
    st.markdown("### 💡 Strategic Intelligence")
    st.markdown(
        f"<p style='color: {t['text_muted']}; font-size: 0.9rem;'>"
        "AI-generated insights explaining the business impact of each metric. "
        "Click on any metric to explore root causes and recommended actions."
        "</p>",
        unsafe_allow_html=True
    )

    # Find metrics that need attention (breaching thresholds)
    critical_kpis = [k for k in kpis if k.threshold_danger and (
        (k.lower_is_better and k.value >= k.threshold_danger) or
        (not k.lower_is_better and k.value <= k.threshold_danger)
    )]

    warning_kpis = [k for k in kpis if k.threshold_warning and k not in critical_kpis and (
        (k.lower_is_better and k.value >= k.threshold_warning) or
        (not k.lower_is_better and k.value <= k.threshold_warning)
    )]

    # Render insights for critical metrics first
    if critical_kpis:
        st.markdown(f"#### 🚨 Critical Alerts ({len(critical_kpis)})")
        for kpi in critical_kpis:
            insight = generate_so_what_insight(kpi)
            render_so_what_panel(insight, kpi.label)

    if warning_kpis:
        st.markdown(f"#### ⚠️ Elevated Risk ({len(warning_kpis)})")
        for kpi in warning_kpis:
            insight = generate_so_what_insight(kpi)
            render_so_what_panel(insight, kpi.label)

    # Show positive performers
    healthy_kpis = [k for k in kpis if k not in critical_kpis and k not in warning_kpis]
    if healthy_kpis:
        with st.expander(f"✅ Healthy Metrics ({len(healthy_kpis)})", expanded=False):
            for kpi in healthy_kpis:
                st.markdown(f'''
                <div style="
                    background: {t['success']}10;
                    border-left: 3px solid {t['success']};
                    padding: 0.75rem 1rem;
                    margin-bottom: 0.5rem;
                    border-radius: 0 8px 8px 0;
                ">
                    <strong style="color: {t['text']};">{kpi.label}</strong>
                    <span style="color: {t['text_muted']}; margin-left: 0.5rem;">
                        {kpi.value}{kpi.unit} - Within target range
                    </span>
                </div>
                ''', unsafe_allow_html=True)


def _render_scenario_simulator(kpis: List[KPIMetric], t: dict):
    """Render Flight Simulator for scenario analysis."""
    st.markdown("### 🎮 Risk Scenario Simulator")
    st.markdown(
        f"<p style='color: {t['text_muted']}; font-size: 0.9rem;'>"
        "Explore 'what-if' scenarios to understand potential impacts and prepare contingency plans. "
        "Adjust parameters to simulate various market conditions and stress scenarios."
        "</p>",
        unsafe_allow_html=True
    )

    # Select metric to simulate
    selected_metric_id = st.selectbox(
        "Select Metric to Simulate",
        options=[k.id for k in kpis],
        format_func=lambda x: next((k.label for k in kpis if k.id == x), x)
    )

    selected_metric = next((k for k in kpis if k.id == selected_metric_id), kpis[0])

    render_flight_simulator(selected_metric)

    # Scenario presets
    st.markdown("#### 📋 Pre-built Stress Scenarios")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f'''
        <div style="
            background: {t['card']};
            border: 1px solid {t['border']};
            border-radius: 12px;
            padding: 1rem;
        ">
            <div style="font-size: 1.25rem; margin-bottom: 0.5rem;">🌪️</div>
            <div style="font-weight: 600; color: {t['text']}; margin-bottom: 0.25rem;">
                OJK Stress Test 2024
            </div>
            <div style="font-size: 0.8rem; color: {t['text_muted']};">
                Regulatory stress scenario with GDP -5%, NPL +3%
            </div>
        </div>
        ''', unsafe_allow_html=True)

    with col2:
        st.markdown(f'''
        <div style="
            background: {t['card']};
            border: 1px solid {t['border']};
            border-radius: 12px;
            padding: 1rem;
        ">
            <div style="font-size: 1.25rem; margin-bottom: 0.5rem;">📉</div>
            <div style="font-weight: 600; color: {t['text']}; margin-bottom: 0.25rem;">
                Commodity Crash
            </div>
            <div style="font-size: 0.8rem; color: {t['text_muted']};">
                Mining & Palm Oil sector shock scenario
            </div>
        </div>
        ''', unsafe_allow_html=True)

    with col3:
        st.markdown(f'''
        <div style="
            background: {t['card']};
            border: 1px solid {t['border']};
            border-radius: 12px;
            padding: 1rem;
        ">
            <div style="font-size: 1.25rem; margin-bottom: 0.5rem;">💹</div>
            <div style="font-weight: 600; color: {t['text']}; margin-bottom: 0.25rem;">
                Rate Normalization
            </div>
            <div style="font-size: 0.8rem; color: {t['text_muted']};">
                BI Rate increase +150bps impact analysis
            </div>
        </div>
        ''', unsafe_allow_html=True)


def _render_detailed_analytics(kpis: List[KPIMetric], t: dict):
    """Render detailed analytics with trends and comparisons."""
    st.markdown("### 📈 Detailed Performance Analytics")

    # KPI comparison table
    st.markdown("#### KPI Performance Summary")

    # Build comparison data
    table_html = f'''
    <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem;">
        <thead>
            <tr style="background: {t['bg_secondary']};">
                <th style="padding: 0.75rem; text-align: left; color: {t['text_muted']}; font-weight: 600; text-transform: uppercase; font-size: 0.7rem; letter-spacing: 0.05em;">Metric</th>
                <th style="padding: 0.75rem; text-align: right; color: {t['text_muted']}; font-weight: 600; text-transform: uppercase; font-size: 0.7rem;">Current</th>
                <th style="padding: 0.75rem; text-align: right; color: {t['text_muted']}; font-weight: 600; text-transform: uppercase; font-size: 0.7rem;">Target</th>
                <th style="padding: 0.75rem; text-align: right; color: {t['text_muted']}; font-weight: 600; text-transform: uppercase; font-size: 0.7rem;">Variance</th>
                <th style="padding: 0.75rem; text-align: center; color: {t['text_muted']}; font-weight: 600; text-transform: uppercase; font-size: 0.7rem;">Trend</th>
                <th style="padding: 0.75rem; text-align: center; color: {t['text_muted']}; font-weight: 600; text-transform: uppercase; font-size: 0.7rem;">Status</th>
            </tr>
        </thead>
        <tbody>
    '''

    for kpi in kpis:
        # Calculate variance
        if kpi.target:
            variance = kpi.value - kpi.target
            variance_pct = (variance / kpi.target * 100) if kpi.target != 0 else 0

            if kpi.lower_is_better:
                variance_color = t['success'] if variance <= 0 else t['danger']
            else:
                variance_color = t['success'] if variance >= 0 else t['danger']
        else:
            variance = 0
            variance_pct = 0
            variance_color = t['text_muted']

        # Trend icon
        trend_icon = "↑" if kpi.trend_direction == TrendDirection.UP else "↓" if kpi.trend_direction == TrendDirection.DOWN else "→"

        # Status badge
        if kpi.threshold_danger:
            if (kpi.lower_is_better and kpi.value >= kpi.threshold_danger) or \
               (not kpi.lower_is_better and kpi.value <= kpi.threshold_danger):
                status = "CRITICAL"
                status_bg = t['danger']
            elif kpi.threshold_warning and ((kpi.lower_is_better and kpi.value >= kpi.threshold_warning) or \
                 (not kpi.lower_is_better and kpi.value <= kpi.threshold_warning)):
                status = "WARNING"
                status_bg = t['warning']
            else:
                status = "NORMAL"
                status_bg = t['success']
        else:
            status = "NORMAL"
            status_bg = t['success']

        table_html += f'''
            <tr style="border-bottom: 1px solid {t['border']};">
                <td style="padding: 0.75rem; color: {t['text']}; font-weight: 500;">{kpi.label}</td>
                <td style="padding: 0.75rem; text-align: right; color: {t['text']}; font-weight: 600;">{kpi.value:.2f}{kpi.unit}</td>
                <td style="padding: 0.75rem; text-align: right; color: {t['text_muted']};">{kpi.target:.2f}{kpi.unit if kpi.target else '-'}</td>
                <td style="padding: 0.75rem; text-align: right; color: {variance_color}; font-weight: 500;">{'+' if variance > 0 else ''}{variance:.2f}{kpi.unit}</td>
                <td style="padding: 0.75rem; text-align: center; font-size: 1.1rem;">{trend_icon}</td>
                <td style="padding: 0.75rem; text-align: center;">
                    <span style="background: {status_bg}20; color: {status_bg}; padding: 0.2rem 0.5rem; border-radius: 8px; font-size: 0.65rem; font-weight: 600;">{status}</span>
                </td>
            </tr>
        '''

    table_html += '''
        </tbody>
    </table>
    '''

    st.markdown(table_html, unsafe_allow_html=True)

    # Data Lineage - Full detail
    st.markdown("#### 🔍 Data Governance & Lineage")

    lineage = DataLineage(
        source_system="Core Banking System (T24 Temenos)",
        extraction_time=datetime.now() - timedelta(minutes=15),
        transformation="ETL Pipeline v3.2 (Airflow) → Snowflake DWH → dbt Models → AURIX Analytics Layer",
        quality_score=0.97,
        owner="Chief Risk Officer (CRO) Office",
        refresh_frequency="Near Real-time (15-minute intervals)"
    )
    render_data_lineage(lineage, compact=False)


def _render_reports_section(kpis: List[KPIMetric], t: dict):
    """Render reports and export section."""
    st.markdown("### 📑 Executive Reports")
    st.markdown(
        f"<p style='color: {t['text_muted']}; font-size: 0.9rem;'>"
        "Generate presentation-ready reports for Board meetings, Risk Committee, and regulatory submissions."
        "</p>",
        unsafe_allow_html=True
    )

    # Export panel
    render_export_panel(
        title="Executive Dashboard Report",
        metrics=kpis
    )

    st.markdown("#### 📋 Available Report Templates")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f'''
        <div style="background: {t['card']}; border: 1px solid {t['border']}; border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.75rem;">
                <span style="font-size: 1.5rem;">📊</span>
                <div style="font-weight: 600; color: {t['text']};">Board Risk Report</div>
            </div>
            <div style="font-size: 0.85rem; color: {t['text_muted']}; margin-bottom: 0.75rem;">
                Comprehensive risk overview for Board of Directors with key metrics, trends, and strategic recommendations.
            </div>
            <div style="font-size: 0.7rem; color: {t['accent']};">
                📄 20-25 slides | ⏱️ ~2 min to generate
            </div>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown(f'''
        <div style="background: {t['card']}; border: 1px solid {t['border']}; border-radius: 12px; padding: 1.25rem;">
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.75rem;">
                <span style="font-size: 1.5rem;">📈</span>
                <div style="font-weight: 600; color: {t['text']};">KRI Dashboard Summary</div>
            </div>
            <div style="font-size: 0.85rem; color: {t['text_muted']}; margin-bottom: 0.75rem;">
                One-page executive summary of all Key Risk Indicators with threshold breaches highlighted.
            </div>
            <div style="font-size: 0.7rem; color: {t['accent']};">
                📄 1 page | ⏱️ ~30 sec to generate
            </div>
        </div>
        ''', unsafe_allow_html=True)

    with col2:
        st.markdown(f'''
        <div style="background: {t['card']}; border: 1px solid {t['border']}; border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.75rem;">
                <span style="font-size: 1.5rem;">🏛️</span>
                <div style="font-weight: 600; color: {t['text']};">OJK Regulatory Report</div>
            </div>
            <div style="font-size: 0.85rem; color: {t['text_muted']}; margin-bottom: 0.75rem;">
                Pre-formatted report following OJK submission guidelines for monthly risk reporting.
            </div>
            <div style="font-size: 0.7rem; color: {t['accent']};">
                📄 Standard format | ⏱️ ~1 min to generate
            </div>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown(f'''
        <div style="background: {t['card']}; border: 1px solid {t['border']}; border-radius: 12px; padding: 1.25rem;">
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.75rem;">
                <span style="font-size: 1.5rem;">⚠️</span>
                <div style="font-weight: 600; color: {t['text']};">Incident Alert Report</div>
            </div>
            <div style="font-size: 0.85rem; color: {t['text_muted']}; margin-bottom: 0.75rem;">
                Detailed analysis of threshold breaches with root cause analysis and remediation timeline.
            </div>
            <div style="font-size: 0.7rem; color: {t['accent']};">
                📄 3-5 pages | ⏱️ ~45 sec to generate
            </div>
        </div>
        ''', unsafe_allow_html=True)

    # Schedule reports
    st.markdown("#### ⏰ Scheduled Reports")
    st.info("💡 Configure automated report generation and distribution to stakeholders. Reports can be scheduled daily, weekly, or on-demand.")


def _render_predictive_alerts(kpis: List[KPIMetric], t: dict):
    """Render predictive alerts with AI-powered breach forecasting."""
    st.markdown("### 🔮 Predictive KPI Alerts")
    st.markdown(
        f"<p style='color: {t['text_muted']}; font-size: 0.9rem;'>"
        "AI-powered forecasting predicts potential threshold breaches 30-90 days ahead, "
        "enabling proactive risk management and early intervention."
        "</p>",
        unsafe_allow_html=True
    )

    # Initialize predictive engine
    engine = PredictiveKPIEngine()

    # Generate time series for each KPI
    forecasts = []
    breach_predictions = []

    for kpi in kpis:
        # Create time series from KPI
        time_series = _create_time_series_from_kpi(kpi)

        # Forecast
        forecast = engine.forecast_kpi(time_series, horizon_days=90)
        forecasts.append((kpi, forecast))

        # Check for breach
        breach = engine.predict_breach(forecast)
        if breach:
            breach_predictions.append((kpi, breach))

    # Alert Panel
    if breach_predictions:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg, {t['danger']}15 0%, {t['warning']}10 100%);
                    border:2px solid {t['danger']}50; border-radius:16px; padding:1.5rem; margin:1rem 0;">
            <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:1rem;">
                <span style="font-size:2rem;">🚨</span>
                <div>
                    <div style="font-size:1.25rem; font-weight:700; color:{t['danger']};">
                        {len(breach_predictions)} Predicted Breach{'es' if len(breach_predictions) > 1 else ''}
                    </div>
                    <div style="color:{t['text_muted']};">AI forecasting detected potential threshold violations</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Individual breach alerts
        for kpi, breach in breach_predictions:
            urgency_color = t['danger'] if breach.days_to_breach <= 30 else t['warning'] if breach.days_to_breach <= 60 else t['accent']

            st.markdown(f"""
            <div style="background:{t['card']}; border:1px solid {urgency_color}; border-left:4px solid {urgency_color};
                        border-radius:0 12px 12px 0; padding:1rem; margin:0.75rem 0;">
                <div style="display:flex; justify-content:space-between; align-items:start;">
                    <div style="flex:1;">
                        <div style="font-weight:700; color:{t['text']}; font-size:1.1rem;">{kpi.label}</div>
                        <div style="color:{t['text_muted']}; margin:0.5rem 0;">
                            Current: <strong>{kpi.value:.2f}{kpi.unit}</strong> →
                            Projected: <strong style="color:{urgency_color};">{breach.projected_value:.2f}{kpi.unit}</strong>
                        </div>
                        <div style="color:{t['text_muted']}; font-size:0.85rem;">
                            Threshold: {breach.threshold_type} at {breach.threshold_value:.2f}{kpi.unit}
                        </div>
                    </div>
                    <div style="text-align:center; padding:0 1rem;">
                        <div style="font-size:2rem; font-weight:800; color:{urgency_color};">{breach.days_to_breach}</div>
                        <div style="font-size:0.75rem; color:{t['text_muted']};">days until breach</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="background:{urgency_color}; color:white; padding:0.5rem 1rem; border-radius:8px; font-weight:600;">
                            {breach.confidence*100:.0f}% Confidence
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Show recommended actions
            if breach.recommended_actions:
                with st.expander(f"📋 Recommended Actions for {kpi.label}"):
                    for i, action in enumerate(breach.recommended_actions, 1):
                        st.markdown(f"{i}. {action}")
    else:
        st.success("✅ No predicted breaches in the next 90 days. All KPIs projected to remain within thresholds.")

    # Forecast visualization
    st.markdown("---")
    st.markdown("### 📈 KPI Forecast Projections")

    selected_kpi = st.selectbox(
        "Select KPI to visualize forecast",
        options=[k.id for k in kpis],
        format_func=lambda x: next((k.label for k in kpis if k.id == x), x)
    )

    selected_forecast = next((f for k, f in forecasts if k.id == selected_kpi), None)
    selected_metric = next((k for k in kpis if k.id == selected_kpi), None)

    if selected_forecast and selected_metric:
        # Display forecast chart
        chart_data = {
            "Date": [dp.date.strftime("%Y-%m-%d") for dp in selected_forecast.forecast_points],
            "Forecast": [dp.value for dp in selected_forecast.forecast_points],
            "Upper CI": [dp.upper_bound for dp in selected_forecast.forecast_points],
            "Lower CI": [dp.lower_bound for dp in selected_forecast.forecast_points],
        }

        st.line_chart(chart_data, x="Date", y=["Forecast", "Upper CI", "Lower CI"])

        # Forecast statistics
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)

        with fcol1:
            st.metric("Current Value", f"{selected_metric.value:.2f}{selected_metric.unit}")
        with fcol2:
            end_value = selected_forecast.forecast_points[-1].value if selected_forecast.forecast_points else selected_metric.value
            delta = end_value - selected_metric.value
            st.metric("90-Day Forecast", f"{end_value:.2f}{selected_metric.unit}", f"{delta:+.2f}")
        with fcol3:
            st.metric("Trend Direction", selected_forecast.trend_direction.upper())
        with fcol4:
            st.metric("Model Accuracy", f"{selected_forecast.model_accuracy*100:.1f}%")

    # What-If Analysis
    st.markdown("---")
    st.markdown("### 🎯 What-If Scenario Analysis")
    st.caption("Simulate how different scenarios would affect KPI forecasts")

    wcol1, wcol2 = st.columns(2)

    with wcol1:
        scenario_name = st.selectbox(
            "Select Scenario",
            ["Economic Downturn", "Market Recovery", "Regulatory Tightening", "Custom"]
        )

        if scenario_name == "Economic Downturn":
            impact_desc = "GDP -3%, Unemployment +2%, NPL +1.5%"
        elif scenario_name == "Market Recovery":
            impact_desc = "GDP +2%, Credit Growth +5%, NIM +0.3%"
        elif scenario_name == "Regulatory Tightening":
            impact_desc = "CAR requirement +2%, LCR +10%"
        else:
            impact_desc = "Define custom impact parameters"

        st.info(f"Scenario Impact: {impact_desc}")

    with wcol2:
        impact_factor = st.slider("Impact Severity (%)", -50, 50, 0, 5)
        confidence_level = st.slider("Confidence Level (%)", 80, 99, 95)

    if st.button("Run What-If Analysis", type="primary"):
        with st.spinner("Simulating scenario impact..."):
            # Create what-if scenario
            scenario = WhatIfScenario(
                name=scenario_name,
                description=impact_desc,
                kpi_impacts={kpi.id: impact_factor / 100 for kpi in kpis}
            )

            # Run analysis
            what_if_result = engine.run_what_if_analysis(
                baseline_forecasts=[f for _, f in forecasts],
                scenario=scenario
            )

            st.session_state['what_if_result'] = what_if_result
            st.success("What-If analysis completed")

    # Display What-If results
    if 'what_if_result' in st.session_state:
        result = st.session_state['what_if_result']

        st.markdown("#### Scenario Impact Summary")

        for impact in result.kpi_impacts[:6]:
            baseline = impact.baseline_value
            scenario_val = impact.scenario_value
            change = scenario_val - baseline
            change_pct = (change / baseline * 100) if baseline != 0 else 0

            impact_color = t['danger'] if change_pct < -5 else t['warning'] if change_pct < 0 else t['success']

            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                        padding:0.5rem; border-bottom:1px solid {t['border']};">
                <span style="color:{t['text']}; font-weight:500;">{impact.kpi_id.upper()}</span>
                <span style="color:{t['text_muted']};">Baseline: {baseline:.2f}</span>
                <span style="color:{t['text_muted']};">Scenario: {scenario_val:.2f}</span>
                <span style="color:{impact_color}; font-weight:600;">{change_pct:+.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

    # Executive Summary Generation
    st.markdown("---")
    st.markdown("### 📝 AI-Generated Executive Summary")

    if st.button("Generate Executive Summary", use_container_width=True):
        with st.spinner("AI generating executive summary..."):
            summary = _generate_executive_summary(kpis, breach_predictions, t)
            st.session_state['exec_summary'] = summary

    if 'exec_summary' in st.session_state:
        st.markdown(f"""
        <div style="background:{t['card']}; border:1px solid {t['border']}; border-radius:12px; padding:1.5rem; margin:1rem 0;">
            <div style="font-weight:700; color:{t['text']}; margin-bottom:1rem; font-size:1.1rem;">
                Executive Risk Summary - {datetime.now().strftime('%B %Y')}
            </div>
            <div style="color:{t['text']}; line-height:1.6;">
                {st.session_state['exec_summary']}
            </div>
        </div>
        """, unsafe_allow_html=True)


def _create_time_series_from_kpi(kpi: KPIMetric) -> KPITimeSeries:
    """Create a time series from KPI metric for forecasting."""
    from datetime import date

    # Generate mock historical data (12 months)
    today = date.today()
    historical_dates = []
    historical_values = []

    base_value = kpi.value
    trend = 0.02 if kpi.trend_direction == TrendDirection.UP else -0.02 if kpi.trend_direction == TrendDirection.DOWN else 0

    for i in range(12, 0, -1):
        hist_date = today - timedelta(days=30 * i)
        historical_dates.append(hist_date)

        # Calculate historical value with trend and noise
        time_effect = trend * i
        noise = random.uniform(-0.03, 0.03)
        hist_value = base_value * (1 - time_effect + noise)
        historical_values.append(hist_value)

    # Add current value
    historical_dates.append(today)
    historical_values.append(base_value)

    # Create threshold
    threshold = KPIThreshold(
        warning_level=kpi.threshold_warning or (kpi.target * 1.1 if kpi.target else None),
        danger_level=kpi.threshold_danger or (kpi.target * 1.2 if kpi.target else None),
        direction="upper" if kpi.lower_is_better else "lower"
    )

    return KPITimeSeries(
        kpi_id=kpi.id,
        kpi_name=kpi.label,
        unit=kpi.unit,
        historical_dates=historical_dates,
        historical_values=historical_values,
        threshold=threshold
    )


def _generate_executive_summary(kpis: List[KPIMetric], breach_predictions: list, t: dict) -> str:
    """Generate AI-powered executive summary."""
    # Count status
    critical = sum(1 for k in kpis if k.threshold_danger and (
        (k.lower_is_better and k.value >= k.threshold_danger) or
        (not k.lower_is_better and k.value <= k.threshold_danger)
    ))
    warning = sum(1 for k in kpis if k.threshold_warning and (
        (k.lower_is_better and k.value >= k.threshold_warning) or
        (not k.lower_is_better and k.value <= k.threshold_warning)
    )) - critical
    healthy = len(kpis) - critical - warning

    # Build summary
    summary = f"""
    <strong>Overall Position:</strong> Of {len(kpis)} monitored KPIs, {healthy} are healthy,
    {warning} require attention, and {critical} are in critical status.<br><br>
    """

    if breach_predictions:
        summary += f"""
        <strong>Predictive Alerts:</strong> AI forecasting has identified {len(breach_predictions)}
        potential threshold breach(es) in the next 90 days. Immediate attention is recommended for
        {', '.join(k.label for k, _ in breach_predictions[:3])}.<br><br>
        """

    # Key observations
    summary += "<strong>Key Observations:</strong><ul>"

    for kpi in kpis:
        if kpi.trend_direction == TrendDirection.UP and kpi.lower_is_better:
            summary += f"<li>{kpi.label} showing concerning upward trend (+{kpi.trend_value:.1f}% {kpi.period})</li>"
        elif kpi.trend_direction == TrendDirection.DOWN and not kpi.lower_is_better:
            summary += f"<li>{kpi.label} declining (-{kpi.trend_value:.1f}% {kpi.period}) - monitor closely</li>"

    summary += "</ul>"

    # Recommendations
    summary += """
    <br><strong>Recommended Actions:</strong>
    <ol>
        <li>Conduct deep-dive analysis on critical KPIs to identify root causes</li>
        <li>Implement early intervention measures for predicted breaches</li>
        <li>Schedule Risk Committee review for high-priority items</li>
        <li>Update contingency plans based on scenario analysis results</li>
    </ol>
    """

    return summary
