"""Pydantic models for the Agentic GPU incident-response system."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ReviewModel(BaseModel):
    """Base model that rejects undeclared fields for reviewer-friendly validation."""

    model_config = ConfigDict(extra="forbid")


class IncidentPayload(BaseModel):
    """Normalized incident payload shared across the LangGraph workflow state."""

    model_config = ConfigDict(extra="allow")

    alert_id: str
    node_id: str
    timestamp: str
    alert_type: str
    alert_signal: str
    summary: str


class AlertHistoryEntry(ReviewModel):
    """A single recent alert used by the triage tool."""

    timestamp: str
    alert_type: str
    severity: Literal["P1", "P2", "P3", "P4"]
    resolution: str


class NodeStatus(ReviewModel):
    """Node health snapshot returned by the node-status tool."""

    gpu_count: int
    gpu_utilization: list[float]
    temperature_celsius: list[float]
    ecc_errors_total: int
    memory_used_pct: float
    running_job_count: int
    uptime_hours: float


class XidError(ReviewModel):
    """A structured GPU error event."""

    code: str
    timestamp: str


class GpuErrorDetails(ReviewModel):
    """Detailed GPU diagnostics used by the diagnosis agent."""

    xid_errors: list[XidError]
    ecc_sbe_count: int
    ecc_dbe_count: int
    retired_pages_count: int
    thermal_violations: int


class RunningJob(ReviewModel):
    """A workload currently scheduled on the affected node."""

    job_id: str
    user: str
    gpu_count: int
    runtime_hours: float
    framework: str
    job_type: str
    status_summary: str


class BlastRadius(ReviewModel):
    """Impact estimate for a disruptive remediation action."""

    affected_job_count: int
    affected_user_count: int
    estimated_reschedule_time: str
    alternative_nodes_available: int


class MaintenanceWindow(ReviewModel):
    """Maintenance-window status consulted before disruptive remediation."""

    in_window: bool
    next_window_start: str
    next_window_hours: float


class ToolContext(ReviewModel):
    """All simulated telemetry exposed to the agent tools for a node."""

    alert_history: list[AlertHistoryEntry]
    node_status: NodeStatus
    gpu_error_details: GpuErrorDetails
    running_jobs: list[RunningJob]
    blast_radius: BlastRadius
    maintenance_window: MaintenanceWindow


class RecentAction(ReviewModel):
    """A recent disruptive action remembered for rate-limit safeguards."""

    timestamp: str
    action_type: str
    node_id: str


class CrossIncidentMemory(ReviewModel):
    """Short-term memory shared across the agent workflow."""

    node_patterns: list[str]
    user_patterns: list[str]
    recent_actions: list[RecentAction]


class AlertPrompt(ReviewModel):
    """A single sample alert line that can be fed directly into the workflow."""

    prompt_id: str
    name: str
    prompt_text: str
    expected_behavior: str | None = None


class AlertPromptBundle(ReviewModel):
    """Collection of sample alert prompts for demonstrations and grading."""

    prompts: list[AlertPrompt]


class NodeProfile(ReviewModel):
    """Host-specific simulated telemetry and cross-incident memory."""

    host: str
    summary: str
    tool_context: ToolContext
    cross_incident_memory: CrossIncidentMemory


class DefaultNodeProfile(ReviewModel):
    """Fallback telemetry used when an alert references an unknown host."""

    summary: str
    tool_context: ToolContext
    cross_incident_memory: CrossIncidentMemory


class NodeProfileBundle(ReviewModel):
    """Collection of per-host profiles plus a safe default profile."""

    default_profile: DefaultNodeProfile
    profiles: list[NodeProfile]


class KnownIssue(ReviewModel):
    """Knowledge-base entry used by the diagnosis agent."""

    issue_id: str
    description: str
    severity: Literal["P1", "P2", "P3", "P4"]
    category: Literal["hardware", "software", "capacity", "network", "workload"]
    keywords: list[str]
    typical_resolution: str
    false_positive_rate: float
    confidence_notes: str


class KnowledgeBase(ReviewModel):
    """GPU incident knowledge base with issue definitions and heuristics."""

    issues: list[KnownIssue]


class SeverityGatePolicy(ReviewModel):
    """Policy that decides which severities must continue to diagnosis."""

    diagnosis_required_for: list[Literal["P1", "P2", "P3", "P4"]]
    auto_resolve_for: list[Literal["P1", "P2", "P3", "P4"]]


class BlastRadiusPolicy(ReviewModel):
    """Policy defining when human approval is required because of impact."""

    human_approval_job_threshold: int


class PeakHoursPolicy(ReviewModel):
    """Business-hours policy for disruptive actions."""

    timezone: str
    business_days: list[int]
    start_hour: int
    end_hour: int
    drain_requires_human_approval: bool


class RateLimitPolicy(ReviewModel):
    """Safeguard limiting how often disruptive actions can be recommended."""

    action_type: str
    max_actions: int
    window_minutes: int


class MaintenanceWindowDefaults(ReviewModel):
    """Default configuration for maintenance-window planning."""

    window_duration_hours: float


class PolicyConfig(ReviewModel):
    """Top-level policy bundle used by the workflow's safeguard layer."""

    severity_gate: SeverityGatePolicy
    blast_radius: BlastRadiusPolicy
    peak_hours: PeakHoursPolicy
    rate_limit: RateLimitPolicy
    confidence_threshold: float
    maintenance_window_defaults: MaintenanceWindowDefaults


class RemediationTemplate(ReviewModel):
    """Structured remediation template selected by the remediation agent."""

    action_type: str
    action: str
    disruptive: bool
    steps: list[str]
    estimated_resolution: str
    rollback_plan: str
    approval_guidance: str


class RemediationTemplateBundle(ReviewModel):
    """Collection of allowed remediation templates."""

    templates: list[RemediationTemplate]
