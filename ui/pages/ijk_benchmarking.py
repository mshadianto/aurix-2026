"""
IJK Benchmarking Page for AURIX 2026.
Interactive benchmark dashboard with peer comparison and historical analysis.
"""

import streamlit as st
from typing import Dict, Optional
from datetime import datetime

from ui.styles.css_builder import get_current_theme
from ui.components import render_page_header, render_footer, render_alert
from modules.ijk_benchmarking import (
    IJKBenchmarkEngine,
    OJKDataFetcher,
    HistoricalTrendAnalyzer,
    PeerRankingEngine,
    InstitutionType,
    BenchmarkStatus,
    MetricCategory,
    BANKING_BENCHMARKS_2024,
    generate_sample_entity_metrics
)


def render():
    """Render the IJK Benchmarking page."""
    t = get_current_theme()

    render_page_header(
        "IJK Industry Benchmarking",
        "Compare financial metrics against Indonesian industry standards and peers"
    )

    # Tabs for different analysis types
    tab1, tab2, tab3, tab4 = st.tabs([
        "Benchmark Overview",
        "Peer Comparison",
        "Historical Trends",
        "Detailed Analysis"
    ])

    with tab1:
        _render_benchmark_tab(t)

    with tab2:
        _render_peer_comparison_tab(t)

    with tab3:
        _render_historical_tab(t)

    with tab4:
        _render_detailed_tab(t)

    render_footer()


def _render_benchmark_tab(t: dict):
    """Render benchmark overview tab."""
    st.markdown("### Industry Benchmark Comparison")
    st.caption("Compare your metrics against OJK industry averages")

    # Entity configuration
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Entity Configuration**")
        entity_name = st.text_input("Entity Name", value="Bank XYZ")
        entity_id = st.text_input("Entity ID", value="BANK-001")

    with col2:
        st.markdown("**Institution Type**")
        inst_type_options = {
            "BUKU 4 (Core Capital > 30T)": InstitutionType.BANK_BUKU4,
            "BUKU 3 (Core Capital 5-30T)": InstitutionType.BANK_BUKU3,
            "BUKU 2 (Core Capital 1-5T)": InstitutionType.BANK_BUKU2,
            "BUKU 1 (Core Capital < 1T)": InstitutionType.BANK_BUKU1,
        }
        selected_type = st.selectbox("Select Category", list(inst_type_options.keys()))
        institution_type = inst_type_options[selected_type]
        period = st.selectbox("Reporting Period", ["2024-Q4", "2024-Q3", "2024-Q2", "2024-Q1"])

    # Metric inputs
    st.markdown("---")
    st.markdown("**Enter Your Metrics**")

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)

    with mcol1:
        car = st.number_input("CAR (%)", 0.0, 50.0, 18.5, 0.1)
        roa = st.number_input("ROA (%)", -5.0, 10.0, 1.9, 0.1)

    with mcol2:
        npl = st.number_input("NPL Gross (%)", 0.0, 15.0, 3.2, 0.1)
        roe = st.number_input("ROE (%)", -10.0, 50.0, 12.5, 0.1)

    with mcol3:
        ldr = st.number_input("LDR (%)", 0.0, 150.0, 88.0, 0.5)
        nim = st.number_input("NIM (%)", 0.0, 15.0, 4.8, 0.1)

    with mcol4:
        bopo = st.number_input("BOPO (%)", 0.0, 100.0, 82.0, 0.5)
        casa = st.number_input("CASA Ratio (%)", 0.0, 100.0, 52.0, 0.5)

    metrics = {
        "car": car,
        "npl_gross": npl,
        "ldr": ldr,
        "roa": roa,
        "roe": roe,
        "nim": nim,
        "bopo": bopo,
        "casa_ratio": casa
    }

    # Run benchmark
    if st.button("Run Benchmark Analysis", type="primary", use_container_width=True):
        with st.spinner("Analyzing against industry benchmarks..."):
            try:
                engine = IJKBenchmarkEngine()
                summary = engine.benchmark_entity(
                    entity_id=entity_id,
                    entity_name=entity_name,
                    institution_type=institution_type,
                    metrics=metrics,
                    period=period
                )

                st.session_state['benchmark_summary'] = summary
                st.session_state['benchmark_metrics'] = metrics
                st.session_state['benchmark_institution'] = institution_type
                st.success(f"Benchmark analysis completed for {entity_name}")

            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Display results
    if 'benchmark_summary' in st.session_state:
        summary = st.session_state['benchmark_summary']

        st.markdown("---")
        st.markdown("### Benchmark Results")

        # Overall score card
        _render_score_card(summary, t)

        # Category scores
        st.markdown("**Category Scores**")
        ccol1, ccol2, ccol3, ccol4 = st.columns(4)

        with ccol1:
            score = summary.capital_score or 0
            st.metric("Capital Adequacy", f"{score:.0f}%ile", _get_score_delta(score))
        with ccol2:
            score = summary.asset_quality_score or 0
            st.metric("Asset Quality", f"{score:.0f}%ile", _get_score_delta(score))
        with ccol3:
            score = summary.profitability_score or 0
            st.metric("Profitability", f"{score:.0f}%ile", _get_score_delta(score))
        with ccol4:
            score = summary.liquidity_score or 0
            st.metric("Liquidity", f"{score:.0f}%ile", _get_score_delta(score))

        # Strengths and Weaknesses
        st.markdown("---")
        scol1, scol2 = st.columns(2)

        with scol1:
            st.markdown("**Strengths**")
            if summary.strengths:
                for s in summary.strengths:
                    st.markdown(f"""
                    <div style="background:{t['success']}20; border-left:3px solid {t['success']}; padding:0.5rem; margin:0.25rem 0; border-radius:4px;">
                        <span style="color:{t['text']};">&#10003; {s}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No standout strengths identified")

        with scol2:
            st.markdown("**Areas for Improvement**")
            if summary.weaknesses:
                for w in summary.weaknesses:
                    st.markdown(f"""
                    <div style="background:{t['warning']}20; border-left:3px solid {t['warning']}; padding:0.5rem; margin:0.25rem 0; border-radius:4px;">
                        <span style="color:{t['text']};">&#9888; {w}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("No significant weaknesses identified")

        # Individual metric results
        st.markdown("---")
        st.markdown("**Detailed Metric Comparison**")
        _render_metric_results(summary.results, t)


def _render_peer_comparison_tab(t: dict):
    """Render peer comparison tab."""
    st.markdown("### Peer Group Comparison")
    st.caption("See how you rank against peer institutions")

    if 'benchmark_metrics' not in st.session_state:
        st.info("Please run a benchmark analysis first in the Overview tab")
        return

    metrics = st.session_state['benchmark_metrics']
    institution_type = st.session_state['benchmark_institution']
    summary = st.session_state.get('benchmark_summary')

    if st.button("Run Peer Comparison", type="primary", use_container_width=True):
        with st.spinner("Fetching peer data and calculating rankings..."):
            try:
                engine = PeerRankingEngine()
                peer_summary = engine.compare_all_metrics(
                    entity_id=summary.entity_id,
                    entity_name=summary.entity_name,
                    metrics=metrics,
                    institution_type=institution_type
                )

                st.session_state['peer_summary'] = peer_summary
                st.success("Peer comparison completed")

            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Display peer comparison
    if 'peer_summary' in st.session_state:
        peer = st.session_state['peer_summary']

        st.markdown("---")

        # Overall ranking
        st.markdown(f"""
        <div style="background:{t['card']}; border:1px solid {t['border']}; border-radius:12px; padding:1.5rem; margin:1rem 0;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-size:0.8rem; color:{t['text_muted']};">Overall Peer Ranking</div>
                    <div style="font-size:2.5rem; font-weight:700; color:{t['primary']};">#{peer.overall_rank}</div>
                    <div style="color:{t['text_muted']};">of {peer.total_peers} peers</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:0.8rem; color:{t['text_muted']};">Average Percentile</div>
                    <div style="font-size:2rem; font-weight:600; color:{t['accent']};">{peer.avg_percentile:.1f}%</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:0.8rem; color:{t['text_muted']};">Top Quartile Metrics</div>
                    <div style="font-size:1.5rem; font-weight:600; color:{t['success']};">{peer.metrics_in_top_quartile} / {len(peer.rankings)}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Competitive position
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Competitive Strengths**")
            for s in peer.competitive_strengths[:5]:
                st.success(s)

        with col2:
            st.markdown("**Competitive Weaknesses**")
            for w in peer.competitive_weaknesses[:5]:
                st.warning(w)

        # Key differentiators
        if peer.key_differentiators:
            st.markdown("**Key Differentiators**")
            for d in peer.key_differentiators:
                st.info(d)

        # Ranking details by metric
        st.markdown("---")
        st.markdown("**Metric Rankings**")

        for ranking in peer.rankings:
            pct = ranking.percentile
            color = t['success'] if pct >= 75 else t['warning'] if pct >= 50 else t['danger']

            st.markdown(f"""
            <div style="background:{t['card']}; border:1px solid {t['border']}; border-radius:8px; padding:0.75rem; margin:0.5rem 0;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                    <div>
                        <span style="font-weight:600; color:{t['text']};">{ranking.metric_name}</span>
                        <span style="color:{t['text_muted']}; font-size:0.85rem;"> - Your value: {ranking.entity_value:.2f}%</span>
                    </div>
                    <div>
                        <span style="background:{color}; color:white; padding:0.25rem 0.75rem; border-radius:4px; font-weight:600;">
                            #{ranking.rank} of {ranking.total_peers}
                        </span>
                    </div>
                </div>
                <div style="display:flex; gap:2rem; font-size:0.85rem;">
                    <span style="color:{t['text_muted']};">Peer Mean: {ranking.peer_mean:.2f}%</span>
                    <span style="color:{t['text_muted']};">Peer Median: {ranking.peer_median:.2f}%</span>
                    <span style="color:{t['text_muted']};">Gap to Best: {ranking.gap_to_best:+.2f}%</span>
                </div>
                <div style="background:{t['border']}; height:6px; border-radius:3px; margin-top:0.5rem; overflow:hidden;">
                    <div style="width:{pct}%; height:100%; background:{color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Narrative
        st.markdown("**Analysis Summary**")
        st.info(peer.narrative)


def _render_historical_tab(t: dict):
    """Render historical trend analysis tab."""
    st.markdown("### Historical Trend Analysis")
    st.caption("Track performance trends over time")

    if 'benchmark_summary' not in st.session_state:
        st.info("Please run a benchmark analysis first in the Overview tab")
        return

    summary = st.session_state['benchmark_summary']
    institution_type = st.session_state['benchmark_institution']

    # Metric selection
    metric_options = list(BANKING_BENCHMARKS_2024.keys())
    selected_metrics = st.multiselect(
        "Select Metrics to Analyze",
        metric_options,
        default=["car", "npl_gross", "roa"],
        format_func=lambda x: BANKING_BENCHMARKS_2024[x]["metric_name"]
    )

    months = st.slider("Analysis Period (months)", 3, 24, 12)

    if st.button("Analyze Trends", type="primary", use_container_width=True):
        with st.spinner("Fetching historical data and analyzing trends..."):
            try:
                analyzer = HistoricalTrendAnalyzer()
                trends = []

                for metric_id in selected_metrics:
                    metric_name = BANKING_BENCHMARKS_2024[metric_id]["metric_name"]
                    trend = analyzer.analyze_trend(
                        entity_id=summary.entity_id,
                        entity_name=summary.entity_name,
                        metric_id=metric_id,
                        metric_name=metric_name,
                        institution_type=institution_type,
                        months=months
                    )
                    trends.append(trend)

                st.session_state['historical_trends'] = trends
                st.success(f"Trend analysis completed for {len(trends)} metrics")

            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Display trends
    if 'historical_trends' in st.session_state:
        trends = st.session_state['historical_trends']

        st.markdown("---")

        for trend in trends:
            # Trend card
            direction_icon = "&#x2197;" if trend.trend_direction == "up" else "&#x2198;" if trend.trend_direction == "down" else "&#x2192;"
            direction_color = t['success'] if trend.trend_direction == "up" else t['danger'] if trend.trend_direction == "down" else t['text_muted']

            st.markdown(f"""
            <div style="background:{t['card']}; border:1px solid {t['border']}; border-radius:12px; padding:1rem; margin:1rem 0;">
                <div style="display:flex; justify-content:space-between; align-items:start; margin-bottom:1rem;">
                    <div>
                        <div style="font-weight:700; font-size:1.1rem; color:{t['text']};">{trend.metric_name}</div>
                        <div style="color:{t['text_muted']}; font-size:0.85rem;">{len(trend.data_points)} months analyzed</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:1.5rem; color:{direction_color};">{direction_icon}</div>
                        <div style="font-size:0.8rem; color:{t['text_muted']};">Trend: {trend.trend_direction.upper()}</div>
                    </div>
                </div>
                <div style="display:flex; gap:2rem;">
                    <div>
                        <div style="font-size:0.75rem; color:{t['text_muted']};">Trend Strength</div>
                        <div style="font-size:1.25rem; font-weight:600; color:{t['text']};">{trend.trend_strength*100:.0f}%</div>
                    </div>
                    <div>
                        <div style="font-size:0.75rem; color:{t['text_muted']};">Avg Percentile</div>
                        <div style="font-size:1.25rem; font-weight:600; color:{t['text']};">{trend.avg_percentile:.1f}%</div>
                    </div>
                    <div>
                        <div style="font-size:0.75rem; color:{t['text_muted']};">Percentile Change</div>
                        <div style="font-size:1.25rem; font-weight:600; color:{direction_color};">{trend.percentile_change:+.1f}%</div>
                    </div>
                    <div>
                        <div style="font-size:0.75rem; color:{t['text_muted']};">Projected Next</div>
                        <div style="font-size:1.25rem; font-weight:600; color:{t['accent']};">{trend.projected_next_period:.2f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Historical chart
            if trend.data_points:
                chart_data = {
                    "Period": [dp.period for dp in trend.data_points],
                    "Your Value": [dp.value for dp in trend.data_points],
                    "Industry Mean": [dp.industry_mean for dp in trend.data_points]
                }
                st.line_chart(chart_data, x="Period", y=["Your Value", "Industry Mean"])

                # Confidence interval for projection
                if trend.confidence_interval:
                    st.caption(f"Projection 95% CI: {trend.confidence_interval[0]:.2f} - {trend.confidence_interval[1]:.2f}")


def _render_detailed_tab(t: dict):
    """Render detailed analysis tab with radar chart and insights."""
    st.markdown("### Detailed Analysis")
    st.caption("Comprehensive view of your competitive position")

    if 'benchmark_summary' not in st.session_state:
        st.info("Please run a benchmark analysis first in the Overview tab")
        return

    summary = st.session_state['benchmark_summary']

    # Radar chart simulation (using bar chart as approximation)
    st.markdown("**Performance Profile**")

    if summary.results:
        # Prepare radar-like data
        chart_data = {
            "Metric": [r.metric_name[:15] for r in summary.results],
            "Your Percentile": [r.percentile_rank for r in summary.results]
        }
        st.bar_chart(chart_data, x="Metric", y="Your Percentile", horizontal=True)

    # Regulatory compliance
    st.markdown("---")
    st.markdown("**Regulatory Compliance Status**")

    compliant = sum(1 for r in summary.results if r.regulatory_compliant)
    total = len(summary.results)

    rcol1, rcol2 = st.columns([1, 3])
    with rcol1:
        if compliant == total:
            st.success(f"All {total} metrics compliant")
        else:
            st.error(f"{total - compliant} metrics need attention")

    with rcol2:
        for r in summary.results:
            if not r.regulatory_compliant:
                st.markdown(f"""
                <div style="background:{t['danger']}20; border:1px solid {t['danger']}; border-radius:8px; padding:0.75rem; margin:0.5rem 0;">
                    <div style="font-weight:600; color:{t['danger']};">{r.metric_name}</div>
                    <div style="color:{t['text']};">Current: {r.entity_value:.2f}% | Gap: {r.regulatory_gap:.2f}%</div>
                    <div style="color:{t['text_muted']}; font-size:0.85rem;">{r.recommendation}</div>
                </div>
                """, unsafe_allow_html=True)

    # Action items
    st.markdown("---")
    st.markdown("**Recommended Actions**")

    priority_actions = [r for r in summary.results if r.recommendation and r.status in [BenchmarkStatus.CONCERN, BenchmarkStatus.BELOW_AVERAGE]]

    if priority_actions:
        for i, r in enumerate(priority_actions, 1):
            urgency = "HIGH" if r.status == BenchmarkStatus.CONCERN else "MEDIUM"
            urgency_color = t['danger'] if urgency == "HIGH" else t['warning']

            st.markdown(f"""
            <div style="background:{t['card']}; border-left:4px solid {urgency_color}; border-radius:0 8px 8px 0; padding:0.75rem; margin:0.5rem 0;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="font-weight:600; color:{t['text']};">{i}. {r.metric_name}</span>
                        <span style="background:{urgency_color}; color:white; padding:0.15rem 0.5rem; border-radius:4px; font-size:0.7rem; margin-left:0.5rem;">{urgency}</span>
                    </div>
                </div>
                <div style="color:{t['text_muted']}; margin-top:0.25rem;">{r.recommendation}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("No urgent actions required - all metrics performing well")

    # Export option
    st.markdown("---")
    if st.button("Export Full Report", use_container_width=True):
        st.info("Report export functionality - PDF generation available in production deployment")


def _render_score_card(summary, t: dict):
    """Render overall score card."""
    percentile = summary.overall_percentile
    color = t['success'] if percentile >= 75 else t['warning'] if percentile >= 50 else t['danger']

    st.markdown(f"""
    <div style="background:linear-gradient(135deg, {t['card']} 0%, {color}10 100%); border:1px solid {color}40; border-radius:16px; padding:1.5rem; margin:1rem 0;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div style="font-size:0.9rem; color:{t['text_muted']};">Overall Industry Position</div>
                <div style="font-size:3rem; font-weight:800; color:{color};">{percentile:.0f}<span style="font-size:1.5rem;">%ile</span></div>
                <div style="color:{t['text']};">{summary.entity_name} | {summary.institution_type.value.upper()}</div>
            </div>
            <div style="text-align:center; padding:0 2rem;">
                <div style="font-size:2rem; font-weight:700; color:{t['success']};">{summary.metrics_above_average}</div>
                <div style="font-size:0.8rem; color:{t['text_muted']};">Above Average</div>
            </div>
            <div style="text-align:center; padding:0 2rem;">
                <div style="font-size:2rem; font-weight:700; color:{t['danger']};">{summary.metrics_below_average}</div>
                <div style="font-size:0.8rem; color:{t['text_muted']};">Below Average</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:2rem; font-weight:700; color:{t['warning'] if summary.regulatory_concerns > 0 else t['success']};">{summary.regulatory_concerns}</div>
                <div style="font-size:0.8rem; color:{t['text_muted']};">Reg. Concerns</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_metric_results(results, t: dict):
    """Render individual metric comparison results."""
    for r in results:
        status_colors = {
            BenchmarkStatus.EXCELLENT: t['success'],
            BenchmarkStatus.GOOD: t['success'],
            BenchmarkStatus.AVERAGE: t['warning'],
            BenchmarkStatus.BELOW_AVERAGE: t['warning'],
            BenchmarkStatus.CONCERN: t['danger']
        }
        color = status_colors.get(r.status, t['text_muted'])

        st.markdown(f"""
        <div style="background:{t['card']}; border:1px solid {t['border']}; border-radius:8px; padding:0.75rem; margin:0.5rem 0;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div style="flex:1;">
                    <div style="font-weight:600; color:{t['text']};">{r.metric_name}</div>
                    <div style="font-size:0.85rem; color:{t['text_muted']};">{r.insight}</div>
                </div>
                <div style="text-align:center; padding:0 1rem;">
                    <div style="font-size:1.25rem; font-weight:700; color:{t['text']};">{r.entity_value:.2f}%</div>
                    <div style="font-size:0.75rem; color:{t['text_muted']};">Your Value</div>
                </div>
                <div style="text-align:center; padding:0 1rem;">
                    <div style="font-size:1.25rem; font-weight:600; color:{t['text_muted']};">{r.industry_mean:.2f}%</div>
                    <div style="font-size:0.75rem; color:{t['text_muted']};">Industry Mean</div>
                </div>
                <div style="text-align:center; padding:0 1rem;">
                    <div style="font-size:1.25rem; font-weight:700; color:{color};">{r.deviation_from_mean:+.1f}%</div>
                    <div style="font-size:0.75rem; color:{t['text_muted']};">Deviation</div>
                </div>
                <div style="min-width:100px;">
                    <div style="background:{color}; color:white; padding:0.25rem 0.75rem; border-radius:4px; text-align:center; font-weight:600; font-size:0.8rem;">
                        {r.status.value.upper()}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def _get_score_delta(score: float) -> str:
    """Get score delta label."""
    if score >= 75:
        return "Top Quartile"
    elif score >= 50:
        return "Above Average"
    elif score >= 25:
        return "Below Average"
    else:
        return "Needs Attention"
