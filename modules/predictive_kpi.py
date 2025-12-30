"""
Predictive KPI Engine for AURIX 2026.
Time-series forecasting for KPIs with breach prediction and what-if analysis.

Features:
- Time-series forecasting using ARIMA patterns
- KPI breach prediction 30/60/90 days ahead
- Confidence intervals for projections
- What-if scenario impact analysis
- AI-powered root cause correlation
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from pydantic import BaseModel, Field
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# ============================================
# Enums and Constants
# ============================================

class ForecastHorizon(str, Enum):
    """Forecast horizon options."""
    DAYS_30 = "30_days"
    DAYS_60 = "60_days"
    DAYS_90 = "90_days"
    DAYS_180 = "180_days"
    YEAR = "365_days"


class TrendDirection(str, Enum):
    """KPI trend direction."""
    IMPROVING = "improving"
    STABLE = "stable"
    DETERIORATING = "deteriorating"
    VOLATILE = "volatile"


class AlertSeverity(str, Enum):
    """Predictive alert severity."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


# KPI thresholds for breach detection
KPI_THRESHOLDS = {
    "car": {"min": 8.0, "warning": 10.0, "target": 14.0, "direction": "higher"},
    "npl_ratio": {"max": 5.0, "warning": 3.5, "target": 2.0, "direction": "lower"},
    "ldr": {"min": 78.0, "max": 92.0, "target": 85.0, "direction": "optimal"},
    "roa": {"min": 1.25, "warning": 1.5, "target": 2.5, "direction": "higher"},
    "roe": {"min": 8.0, "warning": 10.0, "target": 15.0, "direction": "higher"},
    "nim": {"min": 3.0, "warning": 4.0, "target": 5.0, "direction": "higher"},
    "bopo": {"max": 85.0, "warning": 80.0, "target": 75.0, "direction": "lower"},
    "lcr": {"min": 100.0, "warning": 110.0, "target": 120.0, "direction": "higher"},
    "casa_ratio": {"min": 40.0, "warning": 50.0, "target": 60.0, "direction": "higher"},
}


# ============================================
# Pydantic Models
# ============================================

class KPIDataPoint(BaseModel):
    """Single KPI observation."""
    kpi_id: str = Field(..., description="KPI identifier")
    date: datetime = Field(..., description="Observation date")
    value: float = Field(..., description="KPI value")
    is_actual: bool = Field(default=True, description="Actual vs projected")


class KPIThreshold(BaseModel):
    """Threshold configuration for KPI breach detection."""
    warning_level: Optional[float] = Field(default=None, description="Warning threshold")
    danger_level: Optional[float] = Field(default=None, description="Danger threshold")
    direction: str = Field(default="upper", description="'upper' means breach when above, 'lower' when below")


class KPITimeSeries(BaseModel):
    """Time series data for a KPI."""
    kpi_id: str = Field(..., description="KPI identifier")
    kpi_name: str = Field(..., description="KPI display name")
    unit: str = Field(default="%", description="Unit of measurement")
    historical_dates: List[Any] = Field(default_factory=list, description="Historical observation dates")
    historical_values: List[float] = Field(default_factory=list, description="Historical values")
    threshold: Optional[KPIThreshold] = Field(default=None, description="Threshold configuration")
    data_points: List[KPIDataPoint] = Field(default_factory=list)
    start_date: Optional[datetime] = Field(default=None, description="Series start date")
    end_date: Optional[datetime] = Field(default=None, description="Series end date")
    frequency: str = Field(default="daily", description="Data frequency")


class ForecastResult(BaseModel):
    """Single forecast result."""
    date: datetime = Field(..., description="Forecast date")
    predicted_value: float = Field(..., description="Predicted value")
    lower_bound: float = Field(..., description="Lower confidence bound")
    upper_bound: float = Field(..., description="Upper confidence bound")
    confidence_level: float = Field(default=0.95, description="Confidence level")


class KPIForecast(BaseModel):
    """Complete KPI forecast."""
    forecast_id: str = Field(..., description="Forecast identifier")
    kpi_id: str = Field(..., description="KPI identifier")
    kpi_name: str = Field(..., description="KPI name")

    # Current state
    current_value: float = Field(..., description="Current KPI value")
    current_date: datetime = Field(..., description="Current observation date")

    # Forecast results
    horizon: ForecastHorizon = Field(..., description="Forecast horizon")
    forecasts: List[ForecastResult] = Field(default_factory=list)

    # End of horizon summary
    end_value: float = Field(..., description="Predicted value at end of horizon")
    end_lower: float = Field(..., description="Lower bound at end")
    end_upper: float = Field(..., description="Upper bound at end")

    # Trend analysis
    trend_direction: TrendDirection = Field(..., description="Overall trend")
    trend_strength: float = Field(..., ge=0, le=1, description="Trend strength 0-1")

    # Model info
    model_type: str = Field(default="ARIMA", description="Forecasting model used")
    model_accuracy: float = Field(..., description="Model accuracy (MAPE)")

    generated_at: datetime = Field(default_factory=datetime.now)


class BreachPrediction(BaseModel):
    """KPI breach prediction."""
    prediction_id: str = Field(..., description="Prediction identifier")
    kpi_id: str = Field(..., description="KPI identifier")
    kpi_name: str = Field(..., description="KPI name")

    # Breach details
    threshold_type: str = Field(..., description="min, max, or warning")
    threshold_value: float = Field(..., description="Threshold value")
    current_value: float = Field(..., description="Current value")

    # Prediction
    breach_probability: float = Field(..., ge=0, le=1, description="Probability of breach")
    estimated_breach_date: Optional[datetime] = Field(None, description="Estimated breach date")
    days_to_breach: Optional[int] = Field(None, description="Days until breach")

    # Severity
    severity: AlertSeverity = Field(..., description="Alert severity")

    # Context
    trend_direction: TrendDirection = Field(..., description="Current trend")
    contributing_factors: List[str] = Field(default_factory=list)

    # Recommendation
    recommendation: str = Field(..., description="Recommended action")

    generated_at: datetime = Field(default_factory=datetime.now)


class WhatIfScenario(BaseModel):
    """What-if scenario definition."""
    scenario_id: str = Field(..., description="Scenario identifier")
    scenario_name: str = Field(..., description="Scenario name")
    description: str = Field(..., description="Scenario description")

    # Input adjustments
    adjustments: Dict[str, float] = Field(
        default_factory=dict,
        description="KPI adjustments (absolute or percentage)"
    )
    adjustment_type: str = Field(default="absolute", description="absolute or percentage")

    # Time horizon
    horizon_days: int = Field(default=90, description="Scenario horizon in days")


class WhatIfResult(BaseModel):
    """What-if analysis result."""
    scenario_id: str = Field(..., description="Scenario identifier")
    scenario_name: str = Field(..., description="Scenario name")

    # Baseline vs scenario
    baseline_forecasts: Dict[str, float] = Field(
        default_factory=dict,
        description="Baseline KPI forecasts"
    )
    scenario_forecasts: Dict[str, float] = Field(
        default_factory=dict,
        description="Scenario KPI forecasts"
    )

    # Impact analysis
    impacts: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Impact on each KPI (absolute and percentage)"
    )

    # Risk assessment
    breach_risks_baseline: List[str] = Field(default_factory=list)
    breach_risks_scenario: List[str] = Field(default_factory=list)
    risk_change: str = Field(..., description="Risk level change")

    # Narrative
    narrative: str = Field(..., description="Analysis narrative")

    generated_at: datetime = Field(default_factory=datetime.now)


class PredictiveAlertPanel(BaseModel):
    """Aggregated predictive alerts for dashboard."""
    panel_id: str = Field(..., description="Panel identifier")

    # Alerts summary
    total_alerts: int = Field(default=0)
    critical_alerts: int = Field(default=0)
    warning_alerts: int = Field(default=0)

    # Individual predictions
    breach_predictions: List[BreachPrediction] = Field(default_factory=list)

    # Top risks
    top_risks: List[str] = Field(default_factory=list)

    # Time horizon
    horizon_days: int = Field(default=90)

    generated_at: datetime = Field(default_factory=datetime.now)


# ============================================
# Predictive KPI Engine
# ============================================

class PredictiveKPIEngine:
    """
    Engine for KPI forecasting and breach prediction.
    Uses time-series analysis to predict future KPI values.
    """

    def __init__(self):
        """Initialize the engine."""
        self._forecast_counter = 0
        self._prediction_counter = 0
        self._scenario_counter = 0

    def _generate_forecast_id(self) -> str:
        """Generate unique forecast ID."""
        self._forecast_counter += 1
        return f"FCST-{datetime.now().strftime('%Y%m%d')}-{self._forecast_counter:05d}"

    def _generate_prediction_id(self) -> str:
        """Generate unique prediction ID."""
        self._prediction_counter += 1
        return f"PRED-{datetime.now().strftime('%Y%m%d')}-{self._prediction_counter:05d}"

    def _calculate_trend(
        self,
        values: List[float],
        kpi_id: str
    ) -> Tuple[TrendDirection, float]:
        """
        Calculate trend direction and strength from historical values.
        """
        if len(values) < 3:
            return TrendDirection.STABLE, 0.0

        # Linear regression for trend
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)

        # Calculate R-squared for trend strength
        y_pred = slope * x + intercept
        ss_res = np.sum((values - y_pred) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Determine direction based on slope and KPI type
        kpi_config = KPI_THRESHOLDS.get(kpi_id, {"direction": "higher"})
        direction = kpi_config.get("direction", "higher")

        slope_threshold = 0.01 * np.mean(values)  # 1% of mean

        if abs(slope) < slope_threshold:
            return TrendDirection.STABLE, r_squared

        # Check if trend is good or bad based on KPI direction
        if direction == "higher":
            if slope > slope_threshold:
                return TrendDirection.IMPROVING, r_squared
            else:
                return TrendDirection.DETERIORATING, r_squared
        elif direction == "lower":
            if slope < -slope_threshold:
                return TrendDirection.IMPROVING, r_squared
            else:
                return TrendDirection.DETERIORATING, r_squared
        else:  # optimal
            return TrendDirection.STABLE, r_squared

    def _simple_arima_forecast(
        self,
        values: List[float],
        horizon: int,
        confidence: float = 0.95
    ) -> List[ForecastResult]:
        """
        Simple ARIMA-like forecasting using exponential smoothing.
        Production implementation would use statsmodels or prophet.
        """
        if len(values) < 3:
            # Not enough data - return flat forecast
            last_value = values[-1] if values else 0
            return [
                ForecastResult(
                    date=datetime.now() + timedelta(days=i+1),
                    predicted_value=last_value,
                    lower_bound=last_value * 0.9,
                    upper_bound=last_value * 1.1,
                    confidence_level=confidence
                )
                for i in range(horizon)
            ]

        # Calculate trend and seasonality components
        n = len(values)
        x = np.arange(n)

        # Fit linear trend
        slope, intercept = np.polyfit(x, values, 1)

        # Calculate residual standard error
        trend_line = slope * x + intercept
        residuals = np.array(values) - trend_line
        std_error = np.std(residuals) if len(residuals) > 1 else np.mean(values) * 0.05

        # Z-score for confidence interval
        from scipy import stats as scipy_stats
        z_score = scipy_stats.norm.ppf((1 + confidence) / 2)

        forecasts = []
        for i in range(horizon):
            future_x = n + i
            predicted = slope * future_x + intercept

            # Widen confidence interval as we go further
            horizon_factor = 1 + (i / horizon) * 0.5
            margin = z_score * std_error * horizon_factor

            forecasts.append(ForecastResult(
                date=datetime.now() + timedelta(days=i+1),
                predicted_value=round(predicted, 2),
                lower_bound=round(predicted - margin, 2),
                upper_bound=round(predicted + margin, 2),
                confidence_level=confidence
            ))

        return forecasts

    def forecast_kpi(
        self,
        time_series: KPITimeSeries,
        horizon: ForecastHorizon = ForecastHorizon.DAYS_90
    ) -> KPIForecast:
        """
        Generate forecast for a KPI.

        Args:
            time_series: Historical KPI data
            horizon: Forecast horizon

        Returns:
            KPIForecast with predictions
        """
        # Extract values
        values = [dp.value for dp in sorted(time_series.data_points, key=lambda x: x.date)]

        if not values:
            raise ValueError("No data points in time series")

        # Determine horizon in days
        horizon_days = {
            ForecastHorizon.DAYS_30: 30,
            ForecastHorizon.DAYS_60: 60,
            ForecastHorizon.DAYS_90: 90,
            ForecastHorizon.DAYS_180: 180,
            ForecastHorizon.YEAR: 365,
        }.get(horizon, 90)

        # Calculate trend
        trend_direction, trend_strength = self._calculate_trend(values, time_series.kpi_id)

        # Generate forecasts
        forecasts = self._simple_arima_forecast(values, horizon_days)

        # Calculate accuracy (using last 20% as test if available)
        if len(values) >= 10:
            train_size = int(len(values) * 0.8)
            train_values = values[:train_size]
            test_values = values[train_size:]

            # Forecast on training data
            test_forecasts = self._simple_arima_forecast(train_values, len(test_values))

            # Calculate MAPE
            mape = np.mean([
                abs(actual - pred.predicted_value) / actual
                for actual, pred in zip(test_values, test_forecasts)
                if actual != 0
            ]) * 100
        else:
            mape = 15.0  # Default assumption

        return KPIForecast(
            forecast_id=self._generate_forecast_id(),
            kpi_id=time_series.kpi_id,
            kpi_name=time_series.kpi_name,
            current_value=values[-1],
            current_date=datetime.now(),
            horizon=horizon,
            forecasts=forecasts,
            end_value=forecasts[-1].predicted_value,
            end_lower=forecasts[-1].lower_bound,
            end_upper=forecasts[-1].upper_bound,
            trend_direction=trend_direction,
            trend_strength=round(trend_strength, 2),
            model_type="Simple ARIMA",
            model_accuracy=round(100 - mape, 2)
        )

    def predict_breach(
        self,
        forecast: KPIForecast,
        threshold_override: Optional[Dict[str, float]] = None
    ) -> Optional[BreachPrediction]:
        """
        Predict if and when a KPI will breach threshold.

        Args:
            forecast: KPI forecast
            threshold_override: Override default thresholds

        Returns:
            BreachPrediction if breach is likely, None otherwise
        """
        kpi_config = KPI_THRESHOLDS.get(forecast.kpi_id, {})
        if threshold_override:
            kpi_config.update(threshold_override)

        if not kpi_config:
            return None

        direction = kpi_config.get("direction", "higher")

        # Determine relevant threshold
        threshold_value = None
        threshold_type = None

        if direction == "higher":
            if "min" in kpi_config:
                threshold_value = kpi_config["min"]
                threshold_type = "min"
        elif direction == "lower":
            if "max" in kpi_config:
                threshold_value = kpi_config["max"]
                threshold_type = "max"
        else:  # optimal
            if "min" in kpi_config and forecast.current_value < kpi_config["min"]:
                threshold_value = kpi_config["min"]
                threshold_type = "min"
            elif "max" in kpi_config and forecast.current_value > kpi_config["max"]:
                threshold_value = kpi_config["max"]
                threshold_type = "max"

        if threshold_value is None:
            return None

        # Check if breach is predicted
        breach_date = None
        days_to_breach = None
        breach_probability = 0.0

        for i, fc in enumerate(forecast.forecasts):
            # Check if forecast crosses threshold
            if threshold_type == "min" and fc.predicted_value < threshold_value:
                if breach_date is None:
                    breach_date = fc.date
                    days_to_breach = i + 1
                # Higher probability if lower bound also breaches
                if fc.lower_bound < threshold_value:
                    breach_probability = max(breach_probability, 0.9)
                else:
                    breach_probability = max(breach_probability, 0.6)
            elif threshold_type == "max" and fc.predicted_value > threshold_value:
                if breach_date is None:
                    breach_date = fc.date
                    days_to_breach = i + 1
                if fc.upper_bound > threshold_value:
                    breach_probability = max(breach_probability, 0.9)
                else:
                    breach_probability = max(breach_probability, 0.6)

        if breach_probability < 0.2:
            # Check warning threshold
            warning_value = kpi_config.get("warning")
            if warning_value:
                for fc in forecast.forecasts:
                    if (threshold_type == "min" and fc.predicted_value < warning_value) or \
                       (threshold_type == "max" and fc.predicted_value > warning_value):
                        breach_probability = 0.3
                        break

        if breach_probability < 0.1:
            return None

        # Determine severity
        if breach_probability >= 0.8:
            severity = AlertSeverity.CRITICAL
        elif breach_probability >= 0.6:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO

        if days_to_breach and days_to_breach <= 30:
            severity = AlertSeverity.EMERGENCY if severity == AlertSeverity.CRITICAL else AlertSeverity.CRITICAL

        # Generate recommendation
        if threshold_type == "min" and direction == "higher":
            recommendation = f"Increase {forecast.kpi_name} through capital injection or profit retention"
        elif threshold_type == "max" and direction == "lower":
            recommendation = f"Reduce {forecast.kpi_name} through improved collection or write-offs"
        else:
            recommendation = f"Monitor {forecast.kpi_name} closely and prepare contingency plans"

        return BreachPrediction(
            prediction_id=self._generate_prediction_id(),
            kpi_id=forecast.kpi_id,
            kpi_name=forecast.kpi_name,
            threshold_type=threshold_type,
            threshold_value=threshold_value,
            current_value=forecast.current_value,
            breach_probability=round(breach_probability, 2),
            estimated_breach_date=breach_date,
            days_to_breach=days_to_breach,
            severity=severity,
            trend_direction=forecast.trend_direction,
            contributing_factors=self._identify_contributing_factors(forecast),
            recommendation=recommendation
        )

    def _identify_contributing_factors(self, forecast: KPIForecast) -> List[str]:
        """Identify factors contributing to trend."""
        factors = []

        if forecast.trend_direction == TrendDirection.DETERIORATING:
            if forecast.kpi_id == "car":
                factors.append("Declining profitability reducing retained earnings")
                factors.append("Risk-weighted asset growth outpacing capital")
            elif forecast.kpi_id == "npl_ratio":
                factors.append("Economic slowdown affecting borrower repayment capacity")
                factors.append("Sector concentration in stressed industries")
            elif forecast.kpi_id == "ldr":
                factors.append("Credit growth exceeding deposit mobilization")
                factors.append("Shifting customer preference to non-deposit products")
            elif forecast.kpi_id == "nim":
                factors.append("Competitive pressure on lending rates")
                factors.append("Increasing cost of funds")

        if not factors:
            factors.append("Multiple macroeconomic factors")
            factors.append("Industry-wide trend patterns")

        return factors

    def run_what_if_analysis(
        self,
        baseline_forecasts: Dict[str, KPIForecast],
        scenario: WhatIfScenario
    ) -> WhatIfResult:
        """
        Run what-if scenario analysis.

        Args:
            baseline_forecasts: Dict of KPI forecasts
            scenario: What-if scenario to analyze

        Returns:
            WhatIfResult with impact analysis
        """
        baseline_values = {}
        scenario_values = {}
        impacts = {}

        for kpi_id, forecast in baseline_forecasts.items():
            baseline_end = forecast.end_value
            baseline_values[kpi_id] = baseline_end

            # Apply scenario adjustment
            adjustment = scenario.adjustments.get(kpi_id, 0)

            if scenario.adjustment_type == "percentage":
                scenario_end = baseline_end * (1 + adjustment / 100)
            else:
                scenario_end = baseline_end + adjustment

            scenario_values[kpi_id] = round(scenario_end, 2)

            # Calculate impact
            abs_impact = scenario_end - baseline_end
            pct_impact = (abs_impact / baseline_end * 100) if baseline_end != 0 else 0

            impacts[kpi_id] = {
                "absolute": round(abs_impact, 2),
                "percentage": round(pct_impact, 2)
            }

        # Identify breach risks
        baseline_risks = []
        scenario_risks = []

        for kpi_id, value in baseline_values.items():
            config = KPI_THRESHOLDS.get(kpi_id, {})
            if "min" in config and value < config["min"]:
                baseline_risks.append(f"{kpi_id} below minimum")
            if "max" in config and value > config["max"]:
                baseline_risks.append(f"{kpi_id} above maximum")

        for kpi_id, value in scenario_values.items():
            config = KPI_THRESHOLDS.get(kpi_id, {})
            if "min" in config and value < config["min"]:
                scenario_risks.append(f"{kpi_id} below minimum")
            if "max" in config and value > config["max"]:
                scenario_risks.append(f"{kpi_id} above maximum")

        # Determine risk change
        if len(scenario_risks) > len(baseline_risks):
            risk_change = "increased"
        elif len(scenario_risks) < len(baseline_risks):
            risk_change = "decreased"
        else:
            risk_change = "unchanged"

        # Generate narrative
        narrative = f"Scenario '{scenario.scenario_name}' analysis complete. "

        if impacts:
            most_impacted = max(impacts.items(), key=lambda x: abs(x[1]["percentage"]))
            narrative += f"Most impacted KPI: {most_impacted[0]} ({most_impacted[1]['percentage']:+.1f}%). "

        narrative += f"Risk level {risk_change} from baseline."

        return WhatIfResult(
            scenario_id=scenario.scenario_id,
            scenario_name=scenario.scenario_name,
            baseline_forecasts={k: round(v, 2) for k, v in baseline_values.items()},
            scenario_forecasts=scenario_values,
            impacts=impacts,
            breach_risks_baseline=baseline_risks,
            breach_risks_scenario=scenario_risks,
            risk_change=risk_change,
            narrative=narrative
        )

    def generate_alert_panel(
        self,
        forecasts: Dict[str, KPIForecast],
        horizon_days: int = 90
    ) -> PredictiveAlertPanel:
        """
        Generate aggregated predictive alert panel.

        Args:
            forecasts: Dict of KPI forecasts
            horizon_days: Alert horizon

        Returns:
            PredictiveAlertPanel for dashboard display
        """
        predictions = []

        for kpi_id, forecast in forecasts.items():
            prediction = self.predict_breach(forecast)
            if prediction:
                predictions.append(prediction)

        # Sort by severity and probability
        severity_order = {
            AlertSeverity.EMERGENCY: 0,
            AlertSeverity.CRITICAL: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.INFO: 3
        }
        predictions.sort(key=lambda x: (severity_order.get(x.severity, 4), -x.breach_probability))

        # Count by severity
        critical_count = sum(1 for p in predictions if p.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY])
        warning_count = sum(1 for p in predictions if p.severity == AlertSeverity.WARNING)

        # Top risks
        top_risks = []
        for p in predictions[:5]:
            if p.days_to_breach:
                top_risks.append(f"{p.kpi_name}: {p.breach_probability*100:.0f}% breach risk in {p.days_to_breach} days")
            else:
                top_risks.append(f"{p.kpi_name}: {p.breach_probability*100:.0f}% breach risk")

        return PredictiveAlertPanel(
            panel_id=f"PANEL-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            total_alerts=len(predictions),
            critical_alerts=critical_count,
            warning_alerts=warning_count,
            breach_predictions=predictions,
            top_risks=top_risks,
            horizon_days=horizon_days
        )


def generate_sample_kpi_timeseries(
    kpi_id: str = "car",
    kpi_name: str = "Capital Adequacy Ratio",
    days: int = 180,
    start_value: float = 16.0,
    trend: str = "deteriorating"
) -> KPITimeSeries:
    """
    Generate sample KPI time series for testing.

    Args:
        kpi_id: KPI identifier
        kpi_name: KPI display name
        days: Number of days of history
        start_value: Starting KPI value
        trend: "improving", "stable", or "deteriorating"

    Returns:
        KPITimeSeries with mock data
    """
    import random

    data_points = []
    current_value = start_value

    # Determine trend factor
    if trend == "improving":
        trend_factor = 0.01  # +1% per month
    elif trend == "deteriorating":
        trend_factor = -0.015  # -1.5% per month
    else:
        trend_factor = 0

    start_date = datetime.now() - timedelta(days=days)

    for i in range(days):
        # Apply trend and random noise
        daily_trend = trend_factor / 30  # Convert monthly to daily
        noise = random.uniform(-0.002, 0.002)  # ±0.2% daily noise

        current_value = current_value * (1 + daily_trend + noise)

        data_points.append(KPIDataPoint(
            kpi_id=kpi_id,
            date=start_date + timedelta(days=i),
            value=round(current_value, 2),
            is_actual=True
        ))

    return KPITimeSeries(
        kpi_id=kpi_id,
        kpi_name=kpi_name,
        unit="%",
        data_points=data_points,
        start_date=start_date,
        end_date=datetime.now(),
        frequency="daily"
    )


# Export
__all__ = [
    # Enums
    "ForecastHorizon",
    "TrendDirection",
    "AlertSeverity",
    # Models
    "KPIDataPoint",
    "KPITimeSeries",
    "ForecastResult",
    "KPIForecast",
    "BreachPrediction",
    "WhatIfScenario",
    "WhatIfResult",
    "PredictiveAlertPanel",
    # Constants
    "KPI_THRESHOLDS",
    # Classes
    "PredictiveKPIEngine",
    # Functions
    "generate_sample_kpi_timeseries",
]
