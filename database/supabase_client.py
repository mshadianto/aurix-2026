"""
Supabase Client for AURIX 2026.
Handles database connections and operations.

Setup:
1. Install: pip install supabase
2. Copy .env.example to .env and set:
   - SUPABASE_URL=https://uzwzundmdlxyitiyhhtn.supabase.co
   - SUPABASE_KEY=your_anon_key
"""

import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def _get_supabase_config():
    """Get Supabase config from app settings or environment."""
    try:
        from app.config import settings
        return settings.database.supabase_url, settings.database.supabase_key
    except ImportError:
        # Fallback to environment variables
        return (
            os.getenv("SUPABASE_URL", "https://uzwzundmdlxyitiyhhtn.supabase.co"),
            os.getenv("SUPABASE_KEY", "")
        )


# Supabase configuration
SUPABASE_URL, SUPABASE_KEY = _get_supabase_config()

# Try to import supabase
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    logger.warning("Supabase not installed. Run: pip install supabase")


class SupabaseClient:
    """Supabase database client for AURIX."""

    _instance: Optional['SupabaseClient'] = None
    _client: Optional[Any] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._connect()

    def _connect(self):
        """Initialize Supabase connection."""
        if not SUPABASE_AVAILABLE:
            logger.error("Supabase package not available")
            return

        if not SUPABASE_KEY:
            logger.warning("SUPABASE_KEY not set. Database operations will fail.")
            return

        try:
            self._client = create_client(SUPABASE_URL, SUPABASE_KEY)
            logger.info(f"Connected to Supabase: {SUPABASE_URL}")
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")

    @property
    def client(self) -> Optional[Any]:
        """Get Supabase client instance."""
        return self._client

    @property
    def is_connected(self) -> bool:
        """Check if connected to Supabase."""
        return self._client is not None

    # ============================================
    # USER OPERATIONS
    # ============================================

    def get_user(self, user_id: str) -> Optional[Dict]:
        """Get user by ID."""
        if not self.is_connected:
            return None

        try:
            result = self._client.table('users').select('*').eq('id', user_id).single().execute()
            return result.data
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None

    def get_users_by_role(self, role: str) -> List[Dict]:
        """Get all users with a specific role."""
        if not self.is_connected:
            return []

        try:
            result = self._client.table('users').select('*').eq('role', role).execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Error getting users by role: {e}")
            return []

    def update_user_role(self, user_id: str, role: str) -> bool:
        """Update user role."""
        if not self.is_connected:
            return False

        try:
            self._client.table('users').update({'role': role}).eq('id', user_id).execute()
            return True
        except Exception as e:
            logger.error(f"Error updating user role: {e}")
            return False

    # ============================================
    # AUDIT ENGAGEMENT OPERATIONS
    # ============================================

    def get_engagements(self, status: Optional[str] = None) -> List[Dict]:
        """Get audit engagements, optionally filtered by status."""
        if not self.is_connected:
            return []

        try:
            query = self._client.table('audit_engagements').select('*')
            if status:
                query = query.eq('status', status)
            result = query.order('created_at', desc=True).execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Error getting engagements: {e}")
            return []

    def create_engagement(self, data: Dict) -> Optional[Dict]:
        """Create a new audit engagement."""
        if not self.is_connected:
            return None

        try:
            result = self._client.table('audit_engagements').insert(data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error creating engagement: {e}")
            return None

    def update_engagement(self, engagement_id: str, data: Dict) -> bool:
        """Update an audit engagement."""
        if not self.is_connected:
            return False

        try:
            self._client.table('audit_engagements').update(data).eq('id', engagement_id).execute()
            return True
        except Exception as e:
            logger.error(f"Error updating engagement: {e}")
            return False

    # ============================================
    # FINDINGS OPERATIONS
    # ============================================

    def get_findings(self, engagement_id: Optional[str] = None, status: Optional[str] = None) -> List[Dict]:
        """Get findings with optional filters."""
        if not self.is_connected:
            return []

        try:
            query = self._client.table('findings').select('*')
            if engagement_id:
                query = query.eq('engagement_id', engagement_id)
            if status:
                query = query.eq('status', status)
            result = query.order('created_at', desc=True).execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Error getting findings: {e}")
            return []

    def create_finding(self, data: Dict) -> Optional[Dict]:
        """Create a new finding."""
        if not self.is_connected:
            return None

        try:
            result = self._client.table('findings').insert(data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error creating finding: {e}")
            return None

    def update_finding_status(self, finding_id: str, status: str, reviewed_by: Optional[str] = None) -> bool:
        """Update finding status."""
        if not self.is_connected:
            return False

        try:
            update_data = {'status': status}
            if reviewed_by:
                update_data['reviewed_by'] = reviewed_by
            self._client.table('findings').update(update_data).eq('id', finding_id).execute()
            return True
        except Exception as e:
            logger.error(f"Error updating finding status: {e}")
            return False

    # ============================================
    # KRI OPERATIONS
    # ============================================

    def get_kri_dashboard(self) -> List[Dict]:
        """Get KRI dashboard data."""
        if not self.is_connected:
            return []

        try:
            result = self._client.table('v_kri_dashboard').select('*').execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Error getting KRI dashboard: {e}")
            return []

    def record_kri_value(self, kri_id: str, value: float, period_date: str, recorded_by: str) -> bool:
        """Record a KRI value."""
        if not self.is_connected:
            return False

        try:
            self._client.table('kri_values').upsert({
                'kri_id': kri_id,
                'value': value,
                'period_date': period_date,
                'recorded_by': recorded_by
            }).execute()
            return True
        except Exception as e:
            logger.error(f"Error recording KRI value: {e}")
            return False

    def get_kri_history(self, kri_id: str, months: int = 12) -> List[Dict]:
        """Get KRI historical values."""
        if not self.is_connected:
            return []

        try:
            result = self._client.table('kri_values')\
                .select('*')\
                .eq('kri_id', kri_id)\
                .order('period_date', desc=True)\
                .limit(months)\
                .execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Error getting KRI history: {e}")
            return []

    # ============================================
    # STRESS TEST OPERATIONS
    # ============================================

    def save_stress_test_result(self, data: Dict) -> Optional[Dict]:
        """Save stress test result."""
        if not self.is_connected:
            return None

        try:
            result = self._client.table('stress_test_results').insert(data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error saving stress test: {e}")
            return None

    def save_monte_carlo_result(self, data: Dict) -> Optional[Dict]:
        """Save Monte Carlo simulation result."""
        if not self.is_connected:
            return None

        try:
            result = self._client.table('monte_carlo_results').insert(data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error saving Monte Carlo result: {e}")
            return None

    def get_stress_scenarios(self, predefined_only: bool = False) -> List[Dict]:
        """Get stress test scenarios."""
        if not self.is_connected:
            return []

        try:
            query = self._client.table('stress_scenarios').select('*')
            if predefined_only:
                query = query.eq('is_predefined', True)
            result = query.execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Error getting stress scenarios: {e}")
            return []

    # ============================================
    # CONTINUOUS AUDIT OPERATIONS
    # ============================================

    def get_ca_alerts(self, status: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get continuous audit alerts."""
        if not self.is_connected:
            return []

        try:
            query = self._client.table('ca_alerts').select('*, ca_rules(*)')
            if status:
                query = query.eq('status', status)
            result = query.order('triggered_at', desc=True).limit(limit).execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Error getting CA alerts: {e}")
            return []

    def update_alert_status(self, alert_id: str, status: str, reviewed_by: str) -> bool:
        """Update alert status."""
        if not self.is_connected:
            return False

        try:
            self._client.table('ca_alerts').update({
                'status': status,
                'reviewed_by': reviewed_by,
                'reviewed_at': datetime.now().isoformat()
            }).eq('id', alert_id).execute()
            return True
        except Exception as e:
            logger.error(f"Error updating alert status: {e}")
            return False

    # ============================================
    # AUDIT LOG OPERATIONS
    # ============================================

    def log_action(self, user_id: str, action: str, entity_type: str,
                   entity_id: Optional[str] = None, details: Optional[Dict] = None) -> bool:
        """Log an audit action."""
        if not self.is_connected:
            return False

        try:
            self._client.table('audit_logs').insert({
                'user_id': user_id,
                'action': action,
                'entity_type': entity_type,
                'entity_id': entity_id,
                'new_values': details
            }).execute()
            return True
        except Exception as e:
            logger.error(f"Error logging action: {e}")
            return False

    def get_audit_logs(self, user_id: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get audit logs."""
        if not self.is_connected:
            return []

        try:
            query = self._client.table('audit_logs').select('*')
            if user_id:
                query = query.eq('user_id', user_id)
            result = query.order('created_at', desc=True).limit(limit).execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Error getting audit logs: {e}")
            return []

    # ============================================
    # LLM ANALYTICS
    # ============================================

    def log_llm_usage(self, user_id: str, provider: str, model: str,
                      prompt_tokens: int, completion_tokens: int,
                      latency_ms: int, feature: str, success: bool = True,
                      error_message: Optional[str] = None) -> bool:
        """Log LLM usage."""
        if not self.is_connected:
            return False

        try:
            self._client.table('llm_logs').insert({
                'user_id': user_id,
                'provider': provider,
                'model': model,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens,
                'latency_ms': latency_ms,
                'feature': feature,
                'success': success,
                'error_message': error_message
            }).execute()
            return True
        except Exception as e:
            logger.error(f"Error logging LLM usage: {e}")
            return False

    def get_llm_stats(self, days: int = 30) -> Dict:
        """Get LLM usage statistics."""
        if not self.is_connected:
            return {}

        try:
            # Get total calls and tokens
            result = self._client.rpc('get_llm_stats', {'days': days}).execute()
            return result.data or {}
        except Exception as e:
            logger.error(f"Error getting LLM stats: {e}")
            return {}


# Singleton instance
def get_supabase() -> SupabaseClient:
    """Get Supabase client instance."""
    return SupabaseClient()


# Quick test
if __name__ == "__main__":
    client = get_supabase()
    print(f"Connected: {client.is_connected}")
    print(f"URL: {SUPABASE_URL}")
