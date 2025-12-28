"""
Risk Habit Scorecard Module for AURIX 2026.
Implements Atomic Risk Culture philosophy - small habits leading to robust risk culture.

Features:
- Individual risk behavior tracking
- Team/department habit scoring
- Finding closure consistency metrics
- Risk reporting timeliness
- Compliance nudge integration
- Gamification hooks

Philosophy:
Based on James Clear's "Atomic Habits" - focusing on:
1. Small, consistent improvements (1% better every day)
2. Habit stacking for risk behaviors
3. Environment design for compliance
4. Identity-based habit formation
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from pydantic import BaseModel, Field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# ============================================
# Enums and Constants
# ============================================

class HabitCategory(str, Enum):
    """Categories of risk habits."""
    FINDING_MANAGEMENT = "finding_management"
    RISK_REPORTING = "risk_reporting"
    CONTROL_TESTING = "control_testing"
    DOCUMENTATION = "documentation"
    COMMUNICATION = "communication"
    TRAINING = "training"
    SELF_ASSESSMENT = "self_assessment"


class HabitFrequency(str, Enum):
    """Expected frequency of habit execution."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


class ScoreLevel(str, Enum):
    """Risk habit score levels."""
    EXCELLENT = "excellent"  # 90-100
    GOOD = "good"  # 75-89
    ADEQUATE = "adequate"  # 60-74
    NEEDS_IMPROVEMENT = "needs_improvement"  # 40-59
    POOR = "poor"  # 0-39


class TrendDirection(str, Enum):
    """Trend direction for habit scores."""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"


# Habit definitions with weights
RISK_HABITS: Dict[str, Dict[str, Any]] = {
    "finding_closure": {
        "name": "Finding Closure Timeliness",
        "category": HabitCategory.FINDING_MANAGEMENT,
        "description": "Close audit findings within agreed timeline",
        "frequency": HabitFrequency.WEEKLY,
        "weight": 1.5,
        "target_compliance_pct": 95.0,
        "atomic_cue": "Review open findings every Monday morning"
    },
    "risk_update": {
        "name": "Risk Register Update",
        "category": HabitCategory.RISK_REPORTING,
        "description": "Update risk register with new/changed risks",
        "frequency": HabitFrequency.WEEKLY,
        "weight": 1.2,
        "target_compliance_pct": 90.0,
        "atomic_cue": "Add risk updates during Friday risk review"
    },
    "kri_monitoring": {
        "name": "KRI Monitoring",
        "category": HabitCategory.RISK_REPORTING,
        "description": "Review and act on KRI breaches",
        "frequency": HabitFrequency.DAILY,
        "weight": 1.3,
        "target_compliance_pct": 98.0,
        "atomic_cue": "Check KRI dashboard first thing each morning"
    },
    "control_testing": {
        "name": "Control Testing Completion",
        "category": HabitCategory.CONTROL_TESTING,
        "description": "Complete scheduled control tests on time",
        "frequency": HabitFrequency.MONTHLY,
        "weight": 1.4,
        "target_compliance_pct": 95.0,
        "atomic_cue": "Schedule control tests at start of each month"
    },
    "workpaper_quality": {
        "name": "Workpaper Documentation",
        "category": HabitCategory.DOCUMENTATION,
        "description": "Complete workpapers with required evidence",
        "frequency": HabitFrequency.WEEKLY,
        "weight": 1.1,
        "target_compliance_pct": 100.0,
        "atomic_cue": "Document findings immediately after testing"
    },
    "issue_escalation": {
        "name": "Timely Issue Escalation",
        "category": HabitCategory.COMMUNICATION,
        "description": "Escalate critical issues within 24 hours",
        "frequency": HabitFrequency.DAILY,
        "weight": 1.5,
        "target_compliance_pct": 100.0,
        "atomic_cue": "Review critical findings before end of day"
    },
    "training_completion": {
        "name": "Training Completion",
        "category": HabitCategory.TRAINING,
        "description": "Complete assigned training modules on time",
        "frequency": HabitFrequency.MONTHLY,
        "weight": 0.8,
        "target_compliance_pct": 95.0,
        "atomic_cue": "Dedicate Friday afternoon for training"
    },
    "self_assessment": {
        "name": "RCSA Participation",
        "category": HabitCategory.SELF_ASSESSMENT,
        "description": "Participate in Risk Control Self-Assessment",
        "frequency": HabitFrequency.QUARTERLY,
        "weight": 1.0,
        "target_compliance_pct": 100.0,
        "atomic_cue": "Block calendar for RCSA at quarter start"
    },
    "incident_reporting": {
        "name": "Incident Reporting",
        "category": HabitCategory.RISK_REPORTING,
        "description": "Report risk incidents within SLA",
        "frequency": HabitFrequency.DAILY,
        "weight": 1.4,
        "target_compliance_pct": 100.0,
        "atomic_cue": "Log incidents immediately when discovered"
    },
    "peer_review": {
        "name": "Peer Review Completion",
        "category": HabitCategory.DOCUMENTATION,
        "description": "Complete peer reviews of workpapers",
        "frequency": HabitFrequency.WEEKLY,
        "weight": 0.9,
        "target_compliance_pct": 90.0,
        "atomic_cue": "Review one peer workpaper each Wednesday"
    }
}


# ============================================
# Pydantic Models
# ============================================

class HabitDefinition(BaseModel):
    """Definition of a risk habit."""
    habit_id: str = Field(..., description="Unique habit identifier")
    name: str = Field(..., description="Habit name")
    category: HabitCategory = Field(..., description="Habit category")
    description: str = Field(..., description="Habit description")
    frequency: HabitFrequency = Field(..., description="Expected frequency")
    weight: float = Field(default=1.0, ge=0, description="Weight for scoring")
    target_compliance_pct: float = Field(..., ge=0, le=100, description="Target compliance %")
    atomic_cue: str = Field(..., description="Atomic habit cue/trigger")


class HabitExecution(BaseModel):
    """Record of a habit execution."""
    execution_id: str = Field(..., description="Unique execution ID")
    habit_id: str = Field(..., description="Habit identifier")
    user_id: str = Field(..., description="User identifier")
    execution_date: date = Field(..., description="Execution date")
    completed: bool = Field(..., description="Whether habit was completed")
    on_time: bool = Field(default=True, description="Whether completed on time")
    quality_score: Optional[float] = Field(None, ge=0, le=100, description="Quality score if applicable")
    notes: Optional[str] = Field(None, description="Execution notes")
    evidence_attached: bool = Field(default=False, description="Whether evidence was attached")


class HabitStreak(BaseModel):
    """Tracking of habit streaks."""
    habit_id: str = Field(..., description="Habit identifier")
    user_id: str = Field(..., description="User identifier")
    current_streak: int = Field(default=0, ge=0, description="Current consecutive completions")
    longest_streak: int = Field(default=0, ge=0, description="Longest streak ever")
    last_execution_date: Optional[date] = Field(None, description="Last execution date")
    total_completions: int = Field(default=0, ge=0, description="Total completions")
    total_expected: int = Field(default=0, ge=0, description="Total expected completions")


class UserHabitScore(BaseModel):
    """Individual habit score for a user."""
    habit_id: str = Field(..., description="Habit identifier")
    habit_name: str = Field(..., description="Habit name")
    category: HabitCategory = Field(..., description="Habit category")
    compliance_rate: float = Field(..., ge=0, le=100, description="Compliance rate %")
    target_rate: float = Field(..., description="Target rate %")
    gap: float = Field(..., description="Gap to target")
    weighted_score: float = Field(..., description="Weighted score contribution")
    streak: int = Field(default=0, description="Current streak")
    trend: TrendDirection = Field(..., description="Trend direction")
    nudge: Optional[str] = Field(None, description="Improvement nudge")


class UserScorecard(BaseModel):
    """Complete scorecard for a user."""
    user_id: str = Field(..., description="User identifier")
    user_name: str = Field(..., description="User name")
    department: str = Field(..., description="Department")
    period_start: date = Field(..., description="Scoring period start")
    period_end: date = Field(..., description="Scoring period end")

    # Overall scores
    overall_score: float = Field(..., ge=0, le=100, description="Overall habit score")
    score_level: ScoreLevel = Field(..., description="Score level")
    overall_trend: TrendDirection = Field(..., description="Overall trend")

    # Category scores
    category_scores: Dict[str, float] = Field(default_factory=dict, description="Scores by category")

    # Individual habits
    habit_scores: List[UserHabitScore] = Field(default_factory=list, description="Individual habit scores")

    # Achievements
    total_streaks: int = Field(default=0, description="Total active streaks")
    longest_streak: int = Field(default=0, description="Longest current streak")
    habits_at_target: int = Field(default=0, description="Habits meeting target")

    # Recommendations
    top_strength: Optional[str] = Field(None, description="Top performing habit")
    priority_improvement: Optional[str] = Field(None, description="Priority habit to improve")
    daily_nudge: Optional[str] = Field(None, description="Daily improvement nudge")

    timestamp: datetime = Field(default_factory=datetime.now)


class TeamScorecard(BaseModel):
    """Scorecard for a team/department."""
    team_id: str = Field(..., description="Team identifier")
    team_name: str = Field(..., description="Team name")
    period_start: date = Field(..., description="Scoring period start")
    period_end: date = Field(..., description="Scoring period end")

    # Team metrics
    team_size: int = Field(..., description="Number of team members")
    average_score: float = Field(..., ge=0, le=100, description="Average team score")
    score_level: ScoreLevel = Field(..., description="Team score level")
    score_std_dev: float = Field(..., description="Score standard deviation")

    # Distribution
    excellent_count: int = Field(default=0, description="Members with excellent scores")
    good_count: int = Field(default=0, description="Members with good scores")
    needs_improvement_count: int = Field(default=0, description="Members needing improvement")

    # Category performance
    category_scores: Dict[str, float] = Field(default_factory=dict, description="Team category scores")
    weakest_category: Optional[str] = Field(None, description="Category needing most improvement")

    # Member scorecards
    member_scorecards: List[UserScorecard] = Field(default_factory=list, description="Individual scorecards")

    # Recommendations
    team_focus_area: Optional[str] = Field(None, description="Team improvement focus")
    top_performer: Optional[str] = Field(None, description="Top performing member")

    timestamp: datetime = Field(default_factory=datetime.now)


class ComplianceNudge(BaseModel):
    """Compliance nudge message."""
    nudge_id: str = Field(..., description="Nudge identifier")
    user_id: str = Field(..., description="Target user")
    habit_id: str = Field(..., description="Related habit")
    nudge_type: str = Field(..., description="Type: reminder, encouragement, streak_alert")
    message: str = Field(..., description="Nudge message")
    priority: str = Field(default="normal", description="Priority: low, normal, high")
    scheduled_time: datetime = Field(..., description="When to deliver")
    delivered: bool = Field(default=False, description="Whether delivered")
    acted_upon: Optional[bool] = Field(None, description="Whether user acted on nudge")


# ============================================
# Risk Habit Scorecard Engine
# ============================================

class RiskHabitEngine:
    """
    Risk Habit Scorecard Engine.
    Tracks and scores risk culture habits based on Atomic Habits principles.
    """

    def __init__(self):
        """Initialize the engine."""
        self.habit_definitions = {
            hid: HabitDefinition(habit_id=hid, **hdata)
            for hid, hdata in RISK_HABITS.items()
        }
        self._nudge_counter = 0

    def _get_score_level(self, score: float) -> ScoreLevel:
        """Determine score level from numeric score."""
        if score >= 90:
            return ScoreLevel.EXCELLENT
        elif score >= 75:
            return ScoreLevel.GOOD
        elif score >= 60:
            return ScoreLevel.ADEQUATE
        elif score >= 40:
            return ScoreLevel.NEEDS_IMPROVEMENT
        else:
            return ScoreLevel.POOR

    def _calculate_trend(
        self,
        current_rate: float,
        previous_rate: Optional[float]
    ) -> TrendDirection:
        """Calculate trend direction."""
        if previous_rate is None:
            return TrendDirection.STABLE

        diff = current_rate - previous_rate
        if diff > 2:
            return TrendDirection.IMPROVING
        elif diff < -2:
            return TrendDirection.DECLINING
        else:
            return TrendDirection.STABLE

    def _generate_nudge_message(
        self,
        habit: HabitDefinition,
        compliance_rate: float,
        streak: int
    ) -> Optional[str]:
        """Generate improvement nudge based on habit status."""
        if compliance_rate >= habit.target_compliance_pct:
            if streak >= 5:
                return f"Great job! You're on a {streak}-execution streak for {habit.name}. Keep it up!"
            return None

        gap = habit.target_compliance_pct - compliance_rate

        if gap > 20:
            return f"Priority focus: {habit.name} needs attention. Try: {habit.atomic_cue}"
        elif gap > 10:
            return f"Improvement opportunity: {habit.name} is {gap:.0f}% below target. Atomic cue: {habit.atomic_cue}"
        else:
            return f"Almost there! {habit.name} is just {gap:.0f}% from target."

    def calculate_habit_score(
        self,
        habit_id: str,
        executions: List[HabitExecution],
        period_days: int = 30
    ) -> Tuple[float, int, float]:
        """
        Calculate score for a single habit.

        Returns: (compliance_rate, current_streak, weighted_score)
        """
        if habit_id not in self.habit_definitions:
            return 0.0, 0, 0.0

        habit = self.habit_definitions[habit_id]

        # Calculate expected executions based on frequency
        if habit.frequency == HabitFrequency.DAILY:
            expected = period_days
        elif habit.frequency == HabitFrequency.WEEKLY:
            expected = period_days // 7
        elif habit.frequency == HabitFrequency.MONTHLY:
            expected = max(1, period_days // 30)
        else:  # QUARTERLY
            expected = max(1, period_days // 90)

        # Count completions
        completed = sum(1 for e in executions if e.completed and e.on_time)
        on_time_completions = sum(1 for e in executions if e.completed and e.on_time)

        # Calculate compliance rate
        compliance_rate = (completed / expected * 100) if expected > 0 else 0
        compliance_rate = min(100, compliance_rate)  # Cap at 100%

        # Calculate streak (simplified - counts consecutive completions from most recent)
        sorted_execs = sorted(executions, key=lambda e: e.execution_date, reverse=True)
        streak = 0
        for exec in sorted_execs:
            if exec.completed:
                streak += 1
            else:
                break

        # Calculate weighted score
        # Score is based on compliance relative to target, weighted by habit weight
        target = habit.target_compliance_pct
        if compliance_rate >= target:
            base_score = 100
        else:
            base_score = (compliance_rate / target) * 100

        weighted_score = base_score * habit.weight

        return compliance_rate, streak, weighted_score

    def generate_user_scorecard(
        self,
        user_id: str,
        user_name: str,
        department: str,
        executions: List[HabitExecution],
        period_start: date,
        period_end: date,
        previous_period_scores: Optional[Dict[str, float]] = None
    ) -> UserScorecard:
        """
        Generate complete scorecard for a user.

        Args:
            user_id: User identifier
            user_name: User name
            department: Department
            executions: List of habit executions
            period_start: Period start date
            period_end: Period end date
            previous_period_scores: Optional previous period scores for trend

        Returns:
            UserScorecard
        """
        period_days = (period_end - period_start).days + 1

        # Filter executions for this user and period
        user_execs = [
            e for e in executions
            if e.user_id == user_id and period_start <= e.execution_date <= period_end
        ]

        # Group by habit
        execs_by_habit: Dict[str, List[HabitExecution]] = defaultdict(list)
        for exec in user_execs:
            execs_by_habit[exec.habit_id].append(exec)

        habit_scores: List[UserHabitScore] = []
        category_totals: Dict[HabitCategory, List[float]] = defaultdict(list)
        total_weighted_score = 0
        total_weight = 0
        streaks = []
        at_target_count = 0
        top_score = 0
        top_habit = None
        worst_score = 100
        worst_habit = None

        for habit_id, habit in self.habit_definitions.items():
            execs = execs_by_habit.get(habit_id, [])
            compliance_rate, streak, weighted_score = self.calculate_habit_score(
                habit_id, execs, period_days
            )

            # Get previous score for trend
            prev_score = previous_period_scores.get(habit_id) if previous_period_scores else None
            trend = self._calculate_trend(compliance_rate, prev_score)

            gap = habit.target_compliance_pct - compliance_rate
            nudge = self._generate_nudge_message(habit, compliance_rate, streak)

            habit_score = UserHabitScore(
                habit_id=habit_id,
                habit_name=habit.name,
                category=habit.category,
                compliance_rate=round(compliance_rate, 1),
                target_rate=habit.target_compliance_pct,
                gap=round(gap, 1),
                weighted_score=round(weighted_score, 2),
                streak=streak,
                trend=trend,
                nudge=nudge
            )
            habit_scores.append(habit_score)

            # Accumulate totals
            total_weighted_score += weighted_score
            total_weight += habit.weight
            category_totals[habit.category].append(compliance_rate)

            if streak > 0:
                streaks.append(streak)

            if compliance_rate >= habit.target_compliance_pct:
                at_target_count += 1

            if compliance_rate > top_score:
                top_score = compliance_rate
                top_habit = habit.name

            if compliance_rate < worst_score:
                worst_score = compliance_rate
                worst_habit = habit.name

        # Calculate overall score
        overall_score = (total_weighted_score / total_weight) if total_weight > 0 else 0

        # Calculate category scores
        category_scores = {}
        for cat, scores in category_totals.items():
            if scores:
                category_scores[cat.value] = round(sum(scores) / len(scores), 1)

        # Determine overall trend
        if previous_period_scores:
            prev_overall = sum(previous_period_scores.values()) / len(previous_period_scores)
            overall_trend = self._calculate_trend(overall_score, prev_overall)
        else:
            overall_trend = TrendDirection.STABLE

        # Generate daily nudge
        if worst_habit and worst_score < 70:
            worst_habit_def = next(
                (h for h in self.habit_definitions.values() if h.name == worst_habit), None
            )
            daily_nudge = f"Today's focus: {worst_habit_def.atomic_cue}" if worst_habit_def else None
        else:
            daily_nudge = "Keep up the great work! Maintain your risk culture habits."

        return UserScorecard(
            user_id=user_id,
            user_name=user_name,
            department=department,
            period_start=period_start,
            period_end=period_end,
            overall_score=round(overall_score, 1),
            score_level=self._get_score_level(overall_score),
            overall_trend=overall_trend,
            category_scores=category_scores,
            habit_scores=habit_scores,
            total_streaks=len([s for s in streaks if s > 0]),
            longest_streak=max(streaks) if streaks else 0,
            habits_at_target=at_target_count,
            top_strength=top_habit,
            priority_improvement=worst_habit if worst_score < 70 else None,
            daily_nudge=daily_nudge
        )

    def generate_team_scorecard(
        self,
        team_id: str,
        team_name: str,
        user_scorecards: List[UserScorecard]
    ) -> TeamScorecard:
        """
        Generate team scorecard from individual scorecards.

        Args:
            team_id: Team identifier
            team_name: Team name
            user_scorecards: List of individual scorecards

        Returns:
            TeamScorecard
        """
        if not user_scorecards:
            return TeamScorecard(
                team_id=team_id,
                team_name=team_name,
                period_start=date.today(),
                period_end=date.today(),
                team_size=0,
                average_score=0,
                score_level=ScoreLevel.POOR,
                score_std_dev=0
            )

        scores = [sc.overall_score for sc in user_scorecards]
        avg_score = sum(scores) / len(scores)

        # Calculate std dev
        variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
        std_dev = variance ** 0.5

        # Count by level
        excellent = sum(1 for sc in user_scorecards if sc.score_level == ScoreLevel.EXCELLENT)
        good = sum(1 for sc in user_scorecards if sc.score_level == ScoreLevel.GOOD)
        needs_imp = sum(1 for sc in user_scorecards if sc.score_level in [ScoreLevel.NEEDS_IMPROVEMENT, ScoreLevel.POOR])

        # Aggregate category scores
        category_totals: Dict[str, List[float]] = defaultdict(list)
        for sc in user_scorecards:
            for cat, score in sc.category_scores.items():
                category_totals[cat].append(score)

        category_scores = {
            cat: round(sum(scores) / len(scores), 1)
            for cat, scores in category_totals.items()
        }

        # Find weakest category
        weakest = min(category_scores.items(), key=lambda x: x[1]) if category_scores else None

        # Find top performer
        top_performer = max(user_scorecards, key=lambda sc: sc.overall_score)

        return TeamScorecard(
            team_id=team_id,
            team_name=team_name,
            period_start=user_scorecards[0].period_start,
            period_end=user_scorecards[0].period_end,
            team_size=len(user_scorecards),
            average_score=round(avg_score, 1),
            score_level=self._get_score_level(avg_score),
            score_std_dev=round(std_dev, 2),
            excellent_count=excellent,
            good_count=good,
            needs_improvement_count=needs_imp,
            category_scores=category_scores,
            weakest_category=weakest[0] if weakest else None,
            member_scorecards=user_scorecards,
            team_focus_area=f"Improve {weakest[0]} (currently at {weakest[1]}%)" if weakest else None,
            top_performer=top_performer.user_name
        )

    def generate_compliance_nudge(
        self,
        user_id: str,
        habit_id: str,
        nudge_type: str = "reminder"
    ) -> ComplianceNudge:
        """Generate a compliance nudge for a user."""
        self._nudge_counter += 1

        habit = self.habit_definitions.get(habit_id)
        if not habit:
            message = "Remember to maintain your risk culture habits today!"
        else:
            if nudge_type == "reminder":
                message = f"Reminder: {habit.atomic_cue}"
            elif nudge_type == "encouragement":
                message = f"You're doing great with {habit.name}! Keep the momentum going."
            elif nudge_type == "streak_alert":
                message = f"Don't break your streak! Complete {habit.name} today."
            else:
                message = f"Risk habit tip: {habit.atomic_cue}"

        return ComplianceNudge(
            nudge_id=f"NUDGE-{datetime.now().strftime('%Y%m%d')}-{self._nudge_counter:04d}",
            user_id=user_id,
            habit_id=habit_id,
            nudge_type=nudge_type,
            message=message,
            priority="high" if nudge_type == "streak_alert" else "normal",
            scheduled_time=datetime.now() + timedelta(hours=1)
        )


def generate_sample_executions(
    user_id: str,
    days: int = 30,
    completion_rate: float = 0.85
) -> List[HabitExecution]:
    """Generate sample habit executions for testing."""
    import random

    executions = []
    exec_counter = 0

    for habit_id, habit_def in RISK_HABITS.items():
        freq = habit_def["frequency"]

        if freq == HabitFrequency.DAILY.value:
            occurrences = days
        elif freq == HabitFrequency.WEEKLY.value:
            occurrences = days // 7
        elif freq == HabitFrequency.MONTHLY.value:
            occurrences = max(1, days // 30)
        else:
            occurrences = 1

        for i in range(occurrences):
            exec_counter += 1
            completed = random.random() < completion_rate
            on_time = completed and random.random() < 0.9

            executions.append(HabitExecution(
                execution_id=f"EXEC-{exec_counter:05d}",
                habit_id=habit_id,
                user_id=user_id,
                execution_date=date.today() - timedelta(days=random.randint(0, days)),
                completed=completed,
                on_time=on_time,
                quality_score=random.uniform(70, 100) if completed else None,
                evidence_attached=completed and random.random() < 0.8
            ))

    return executions


# Export
__all__ = [
    # Enums
    "HabitCategory",
    "HabitFrequency",
    "ScoreLevel",
    "TrendDirection",
    # Models
    "HabitDefinition",
    "HabitExecution",
    "HabitStreak",
    "UserHabitScore",
    "UserScorecard",
    "TeamScorecard",
    "ComplianceNudge",
    # Constants
    "RISK_HABITS",
    # Classes
    "RiskHabitEngine",
    # Functions
    "generate_sample_executions",
]
