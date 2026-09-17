"""
Auditable Pipeline Decision Trace.
Cross-cutting logger providing verifiable transparency into each stage of the summarization pipeline.
"""

from typing import List, Dict, Any, Optional
import time
from pydantic import BaseModel, Field


class DecisionStep(BaseModel):
    step_number: int
    name: str
    description: str
    duration_ms: float
    details: Dict[str, Any] = Field(default_factory=dict)
    status: str = "completed"  # "completed", "warning", "error"


class DecisionTrace:
    """
    Records and formats real-time pipeline execution milestones for UI explainability.
    """

    def __init__(self):
        self.steps: List[DecisionStep] = []
        self._start_time = time.perf_counter()
        self._step_start_time = time.perf_counter()

    def log_step(
        self,
        name: str,
        description: str,
        details: Optional[Dict[str, Any]] = None,
        status: str = "completed"
    ):
        """Record completion of a pipeline milestone."""
        now = time.perf_counter()
        duration_ms = round((now - self._step_start_time) * 1000, 1)
        self._step_start_time = now

        step = DecisionStep(
            step_number=len(self.steps) + 1,
            name=name,
            description=description,
            duration_ms=duration_ms,
            details=details or {},
            status=status
        )
        self.steps.append(step)

    def get_total_duration_ms(self) -> float:
        """Returns total elapsed time across all logged steps."""
        return round((time.perf_counter() - self._start_time) * 1000, 1)

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to structured dictionary for dashboard rendering."""
        return {
            "total_duration_ms": self.get_total_duration_ms(),
            "total_steps": len(self.steps),
            "steps": [s.model_dump() for s in self.steps]
        }

    def format_summary_text(self) -> str:
        """Format an ASCII/Markdown execution summary for reports."""
        lines = [f"### Pipeline Decision Trace (Total: {self.get_total_duration_ms()}ms)"]
        for s in self.steps:
            status_icon = "✓" if s.status == "completed" else "⚠"
            lines.append(f"- **Step {s.step_number}: {s.name}** [{status_icon} in {s.duration_ms}ms]")
            lines.append(f"  *{s.description}*")
            if s.details:
                for k, v in s.details.items():
                    lines.append(f"    • {k}: {v}")
        return "\n".join(lines)
