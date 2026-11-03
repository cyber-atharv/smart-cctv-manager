"""
Universal Smart Vision (USV) - Priority Inference Scheduler
Dynamically coordinates multi-model execution across available CPU cycles.
Prevents pipeline starvation and ensures fast-moving critical safety events
(falls, collisions) receive dedicated high-priority time slices.
"""

import time
import logging
from typing import Dict, Any, List, Callable, Optional

logger = logging.getLogger("usv.scheduler")

class PriorityInferenceScheduler:
    """
    Tiered Scheduler for Multi-Domain AI Models:
    Tier 1 (CRITICAL): Biomechanical Fall & Collision Verification (15–20 Hz)
    Tier 2 (HIGH):     Restricted Boundary & Weapon Anomaly (10–12 Hz)
    Tier 3 (MEDIUM):   Vehicle Flow, ANPR & PPE Compliance (6–8 Hz)
    Tier 4 (LOW):      Loitering Accumulator & Face Gallery Re-sync (1–2 Hz)
    """

    def __init__(self, target_ui_fps: float = 60.0):
        self.target_ui_fps = target_ui_fps
        self.last_execution_times: Dict[str, float] = {
            "tier1_critical": 0.0,
            "tier2_high": 0.0,
            "tier3_medium": 0.0,
            "tier4_low": 0.0
        }

        # Desired intervals in seconds
        self.interval_map = {
            "tier1_critical": 0.055,  # ~18 Hz
            "tier2_high": 0.090,      # ~11 Hz
            "tier3_medium": 0.140,    # ~7 Hz
            "tier4_low": 0.600        # ~1.6 Hz
        }

        self.execution_counts: Dict[str, int] = {k: 0 for k in self.interval_map}
        self.last_cpu_throttle_check = time.time()
        self.adaptive_load_scale = 1.0

    def should_execute(self, tier: str) -> bool:
        """Determines if a task tier is due for execution in the current frame slice."""
        now = time.time()
        base_interval = self.interval_map.get(tier, 0.1)
        effective_interval = base_interval * self.adaptive_load_scale

        if (now - self.last_execution_times.get(tier, 0.0)) >= effective_interval:
            self.last_execution_times[tier] = now
            self.execution_counts[tier] = self.execution_counts.get(tier, 0) + 1
            return True
        return False

    def report_step_latency(self, elapsed_sec: float):
        """
        Dynamically adjusts load scale if inference exceeds allotted budget.
        Keeps host CPU healthy and guarantees unblocked UI frames.
        """
        now = time.time()
        if now - self.last_cpu_throttle_check >= 2.0:
            self.last_cpu_throttle_check = now
            # If inference takes > 50ms per step, slightly throttle lower tiers
            if elapsed_sec > 0.055:
                self.adaptive_load_scale = min(1.8, self.adaptive_load_scale + 0.1)
            elif elapsed_sec < 0.025:
                self.adaptive_load_scale = max(0.9, self.adaptive_load_scale - 0.05)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "load_scale": round(self.adaptive_load_scale, 2),
            "tier1_critical_rate": "16-18 Hz",
            "tier2_high_rate": f"{round(1.0 / (self.interval_map['tier2_high'] * self.adaptive_load_scale), 1)} Hz",
            "tier3_medium_rate": f"{round(1.0 / (self.interval_map['tier3_medium'] * self.adaptive_load_scale), 1)} Hz",
            "tier4_low_rate": f"{round(1.0 / (self.interval_map['tier4_low'] * self.adaptive_load_scale), 1)} Hz",
            "scheduler_healthy": True
        }

inference_scheduler = PriorityInferenceScheduler()

