"""
Stress Tester Page for AURIX 2026.
Monte Carlo simulation and reverse stress testing interface.
"""

import streamlit as st
from typing import Dict, List
from datetime import datetime

from ui.styles.css_builder import get_current_theme
from ui.components import render_page_header, render_footer, render_alert
from modules.stress_tester import (
    MacroStressTester,
    MonteCarloSimulator,
    ReverseStressTester,
    MonteCarloConfig,
    generate_sample_portfolio,
    STRESS_SCENARIOS
)


def render():
    """Render the stress tester page."""
    t = get_current_theme()

    render_page_header(
        "Macro-Financial Stress Tester",
        "Monte Carlo simulation and reverse stress testing for capital adequacy"
    )

    # Tabs for different analysis types
    tab1, tab2, tab3 = st.tabs([
        "Monte Carlo Simulation",
        "Reverse Stress Test",
        "Scenario Analysis"
    ])

    with tab1:
        _render_monte_carlo_tab(t)

    with tab2:
        _render_reverse_stress_tab(t)

    with tab3:
        _render_scenario_tab(t)

    render_footer()


def _render_monte_carlo_tab(t: dict):
    """Render Monte Carlo simulation tab."""
    st.markdown("### Monte Carlo Stress Simulation")
    st.caption("Run thousands of randomized scenarios to assess CAR distribution and breach probability")

    # Configuration
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Simulation Configuration**")
        num_sims = st.slider("Number of Simulations", 100, 5000, 1000, 100)
        seed = st.number_input("Random Seed (optional)", min_value=0, value=42)
        use_seed = st.checkbox("Use fixed seed for reproducibility", value=True)

    with col2:
        st.markdown("**Macro Variable Distributions**")
        bi_rate_std = st.slider("BI Rate Volatility (bps)", 50, 200, 100)
        usdidr_std = st.slider("USD/IDR Volatility (%)", 5, 20, 10)
        gdp_std = st.slider("GDP Growth Volatility (%)", 1, 5, 2)

    # Portfolio input
    st.markdown("---")
    st.markdown("**Portfolio Parameters**")

    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    with pcol1:
        current_car = st.number_input("Current CAR (%)", 10.0, 30.0, 16.5, 0.5)
    with pcol2:
        current_npl = st.number_input("Current NPL (%)", 0.0, 10.0, 2.8, 0.1)
    with pcol3:
        total_assets = st.number_input("Total Assets (IDR B)", 10000, 500000, 120000, 1000)
    with pcol4:
        total_capital = st.number_input("Total Capital (IDR B)", 1000, 50000, 19800, 100)

    # Run simulation
    if st.button("Run Monte Carlo Simulation", type="primary", use_container_width=True):
        with st.spinner("Running simulation... This may take a moment."):
            try:
                # Create portfolio
                portfolio = generate_sample_portfolio()
                portfolio.current_car = current_car
                portfolio.npl_ratio = current_npl
                portfolio.total_assets = total_assets
                portfolio.total_capital = total_capital

                # Configure simulation
                config = MonteCarloConfig(
                    num_simulations=num_sims,
                    seed=seed if use_seed else None,
                    bi_rate_dist=(0, bi_rate_std),
                    usdidr_dist=(0, usdidr_std),
                    gdp_dist=(0, gdp_std)
                )

                # Run simulation
                simulator = MonteCarloSimulator()
                result = simulator.run_simulation(portfolio, config)

                # Store result
                st.session_state['mc_result'] = result

                st.success(f"Simulation completed: {num_sims:,} scenarios analyzed")

            except Exception as e:
                st.error(f"Simulation error: {str(e)}")

    # Display results
    if 'mc_result' in st.session_state:
        result = st.session_state['mc_result']

        st.markdown("---")
        st.markdown("### Simulation Results")

        # Key metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Mean CAR", f"{result.car_mean:.2f}%", f"{result.car_mean - current_car:+.2f}%")
        with m2:
            st.metric("CAR Std Dev", f"{result.car_std:.2f}%")
        with m3:
            color = "inverse" if result.prob_car_below_8 > 0.05 else "normal"
            st.metric("P(CAR < 8%)", f"{result.prob_car_below_8*100:.1f}%")
        with m4:
            st.metric("VaR (99%)", f"IDR {result.var_99:.0f}B")

        # Risk rating
        rating_colors = {
            "LOW": t['success'],
            "MODERATE": t['warning'],
            "HIGH": t['danger'],
            "CRITICAL": t['danger']
        }
        st.markdown(f"""
        <div style="background:{t['card']}; border:1px solid {t['border']}; border-radius:12px; padding:1rem; margin:1rem 0;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-weight:700; color:{t['text']}; font-size:1.1rem;">Risk Assessment</div>
                    <div style="color:{t['text_muted']}; font-size:0.85rem;">Based on {result.num_simulations:,} simulations</div>
                </div>
                <div style="background:{rating_colors.get(result.risk_rating, t['accent'])}; color:white; padding:0.5rem 1.5rem; border-radius:8px; font-weight:700;">
                    {result.risk_rating}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Distribution chart
        st.markdown("**CAR Distribution**")

        import numpy as np
        car_data = result.car_distribution

        # Create histogram data
        hist, bin_edges = np.histogram(car_data, bins=50)

        # Display as bar chart
        chart_data = {
            "CAR Range": [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}%" for i in range(len(hist))],
            "Frequency": hist.tolist()
        }
        st.bar_chart(chart_data, x="CAR Range", y="Frequency")

        # Percentiles table
        st.markdown("**CAR Percentiles**")
        pcol1, pcol2, pcol3, pcol4, pcol5 = st.columns(5)
        pcol1.metric("1st %ile", f"{result.car_percentiles.get('p1', 0):.2f}%")
        pcol2.metric("5th %ile", f"{result.car_percentiles.get('p5', 0):.2f}%")
        pcol3.metric("50th %ile", f"{result.car_percentiles.get('p50', 0):.2f}%")
        pcol4.metric("95th %ile", f"{result.car_percentiles.get('p95', 0):.2f}%")
        pcol5.metric("99th %ile", f"{result.car_percentiles.get('p99', 0):.2f}%")

        # Narrative
        st.markdown("**Analysis Narrative**")
        st.info(result.narrative)


def _render_reverse_stress_tab(t: dict):
    """Render reverse stress testing tab."""
    st.markdown("### Reverse Stress Testing")
    st.caption("Find scenarios that would cause CAR to breach regulatory threshold")

    # Configuration
    col1, col2 = st.columns(2)

    with col1:
        target_car = st.slider("Target CAR Threshold (%)", 6.0, 12.0, 8.0, 0.5)
        max_iterations = st.slider("Max Search Iterations", 50, 500, 100, 50)

    with col2:
        st.markdown("**Current Position**")
        current_car = st.number_input("Current CAR (%)", 10.0, 30.0, 16.5, 0.5, key="rst_car")
        buffer = current_car - target_car
        st.info(f"Current buffer: {buffer:.2f} percentage points")

    if st.button("Run Reverse Stress Test", type="primary", use_container_width=True):
        with st.spinner("Searching for breaking scenarios..."):
            try:
                portfolio = generate_sample_portfolio()
                portfolio.current_car = current_car

                tester = ReverseStressTester()
                result = tester.find_breaking_scenarios(portfolio, target_car, max_iterations)

                st.session_state['rst_result'] = result
                st.success(f"Analysis complete: Found {len(result.breaking_scenarios)} breaking scenarios")

            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Display results
    if 'rst_result' in st.session_state:
        result = st.session_state['rst_result']

        st.markdown("---")
        st.markdown("### Reverse Stress Results")

        # Key vulnerabilities
        if result.key_vulnerabilities:
            st.markdown("**Key Vulnerabilities**")
            for vuln in result.key_vulnerabilities:
                st.warning(vuln)

        # Minimum shocks required
        st.markdown("**Minimum Shock Required to Breach**")
        scol1, scol2, scol3 = st.columns(3)

        with scol1:
            if result.min_bi_rate_shock:
                st.metric("BI Rate Increase", f"+{result.min_bi_rate_shock} bps")
            else:
                st.metric("BI Rate Increase", "N/A")

        with scol2:
            if result.min_usdidr_shock:
                st.metric("USD/IDR Depreciation", f"+{result.min_usdidr_shock}%")
            else:
                st.metric("USD/IDR Depreciation", "N/A")

        with scol3:
            if result.min_credit_loss:
                st.metric("Credit Loss Rate", f"{result.min_credit_loss}%")
            else:
                st.metric("Credit Loss Rate", "N/A")

        # Sensitivity ranking
        if result.sensitivity_ranking:
            st.markdown("**Risk Factor Sensitivity**")
            for factor, score in sorted(result.sensitivity_ranking.items(), key=lambda x: -x[1]):
                pct = score * 100
                st.markdown(f"""
                <div style="margin-bottom:0.5rem;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:0.25rem;">
                        <span style="color:{t['text']};">{factor.replace('_', ' ').title()}</span>
                        <span style="color:{t['text_muted']};">{pct:.1f}%</span>
                    </div>
                    <div style="background:{t['border']}; height:8px; border-radius:4px; overflow:hidden;">
                        <div style="width:{pct}%; height:100%; background:{t['primary']};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Recommendations
        if result.recommendations:
            st.markdown("**Recommendations**")
            for rec in result.recommendations:
                if "IMMEDIATE" in rec or "URGENT" in rec:
                    st.error(rec)
                else:
                    st.info(rec)

        # Narrative
        st.markdown("**Analysis Summary**")
        st.info(result.narrative)


def _render_scenario_tab(t: dict):
    """Render predefined scenario analysis tab."""
    st.markdown("### Predefined Stress Scenarios")
    st.caption("Run Basel IV compliant stress scenarios")

    # Portfolio configuration
    col1, col2 = st.columns(2)
    with col1:
        current_car = st.number_input("Current CAR (%)", 10.0, 30.0, 16.5, 0.5, key="scen_car")
    with col2:
        current_npl = st.number_input("Current NPL (%)", 0.0, 10.0, 2.8, 0.1, key="scen_npl")

    # Scenario selection
    st.markdown("---")
    st.markdown("**Select Scenarios to Run**")

    scenario_ids = list(STRESS_SCENARIOS.keys())
    selected = st.multiselect(
        "Scenarios",
        scenario_ids,
        default=scenario_ids[:3],
        format_func=lambda x: STRESS_SCENARIOS[x].scenario_name
    )

    if st.button("Run Selected Scenarios", type="primary"):
        if not selected:
            st.warning("Please select at least one scenario")
            return

        with st.spinner("Running stress tests..."):
            portfolio = generate_sample_portfolio()
            portfolio.current_car = current_car
            portfolio.npl_ratio = current_npl

            tester = MacroStressTester()
            results = []

            for scenario_id in selected:
                scenario = STRESS_SCENARIOS[scenario_id]
                result = tester.run_stress_test(portfolio, scenario)
                results.append(result)

            st.session_state['scenario_results'] = results
            st.success(f"Completed {len(results)} scenario tests")

    # Display results
    if 'scenario_results' in st.session_state:
        results = st.session_state['scenario_results']

        st.markdown("---")
        st.markdown("### Scenario Results")

        for result in results:
            severity_colors = {
                "MILD": t['success'],
                "MODERATE": t['warning'],
                "SEVERE": t['danger'],
                "EXTREME": t['danger']
            }

            color = severity_colors.get(result.scenario.severity.value.upper(), t['border'])

            st.markdown(f"""
            <div style="background:{t['card']}; border:1px solid {color}; border-left:4px solid {color}; border-radius:12px; padding:1rem; margin-bottom:1rem;">
                <div style="display:flex; justify-content:space-between; align-items:start; margin-bottom:0.5rem;">
                    <div>
                        <div style="font-weight:700; color:{t['text']};">{result.scenario.scenario_name}</div>
                        <div style="font-size:0.8rem; color:{t['text_muted']};">{result.scenario.description}</div>
                    </div>
                    <div style="background:{color}; color:white; padding:0.25rem 0.75rem; border-radius:4px; font-size:0.75rem; font-weight:600;">
                        {result.outcome.value.upper()}
                    </div>
                </div>
                <div style="display:flex; gap:2rem; margin-top:0.75rem;">
                    <div>
                        <div style="font-size:0.7rem; color:{t['text_muted']};">Projected CAR</div>
                        <div style="font-size:1.25rem; font-weight:700; color:{color};">{result.projections.projected_car:.2f}%</div>
                    </div>
                    <div>
                        <div style="font-size:0.7rem; color:{t['text_muted']};">Total Loss</div>
                        <div style="font-size:1.25rem; font-weight:700; color:{t['text']};">IDR {result.total_loss:.0f}B</div>
                    </div>
                    <div>
                        <div style="font-size:0.7rem; color:{t['text_muted']};">Projected NPL</div>
                        <div style="font-size:1.25rem; font-weight:700; color:{t['text']};">{result.projections.projected_npl:.2f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Comparison chart
        st.markdown("**CAR Impact Comparison**")
        chart_data = {
            "Scenario": [r.scenario.scenario_name[:20] for r in results],
            "Projected CAR (%)": [r.projections.projected_car for r in results]
        }
        st.bar_chart(chart_data, x="Scenario", y="Projected CAR (%)")
