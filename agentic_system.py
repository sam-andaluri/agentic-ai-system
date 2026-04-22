"""Agentic GPU incident-response workflow.

System scope:
This project implements a multi-agent GPU cluster incident-response assistant for
academic use. The workflow accepts a single infrastructure alert, reasons across
simulated telemetry and incident memory, and produces a recommendation instead of
executing infrastructure changes directly.

Architecture summary:
The system uses a prompt-driven orchestrator plus three specialized agents:
triage, diagnosis, and remediation. LangGraph manages the agent loop and shared
state, LangChain tools provide controlled access to telemetry, and a deterministic
policy layer enforces safeguards such as blast-radius checks, peak-hours
protection, confidence thresholds, rate limits, and rollback requirements.

Execution summary:
Representative sample alerts and simulated node profiles live in external JSON
files so the Python runtime stays configuration driven. The script supports
single-run CLI execution from sample alerts plus an interactive alert loop that
triggers the same full workflow path for each alert entered.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from models import (
    AlertPromptBundle,
    IncidentPayload,
    KnowledgeBase,
    NodeProfileBundle,
    PolicyConfig,
    RemediationTemplateBundle,
)


class OrchestratorDecision(BaseModel):
    """Structured route choice returned by the orchestrator."""

    route: Literal["triage", "diagnosis", "remediation", "finalize"]
    reason: str = Field(min_length=1)
    audit_entry: str = Field(min_length=1)


class TriageDecision(BaseModel):
    """Structured triage output returned after initial alert review."""

    severity: Literal["P1", "P2", "P3", "P4"]
    category: Literal["hardware", "software", "capacity", "network", "workload"]
    escalate: bool
    auto_resolve: bool
    reasoning: str = Field(min_length=1)
    supporting_evidence: list[str] = Field(default_factory=list)


class DiagnosisDecision(BaseModel):
    """Structured diagnosis result with evidence and confidence."""

    root_cause: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    knowledge_base_matches: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    reasoning: str = Field(min_length=1)


class RemediationDecision(BaseModel):
    """Structured remediation recommendation produced before safeguard enforcement."""

    action_type: str = Field(min_length=1)
    justification: str = Field(min_length=1)
    approval: Literal["AUTO-APPROVED", "HUMAN-REQUIRED"]
    blast_radius_summary: str = Field(min_length=1)
    reasoning: str = Field(min_length=1)


class NodeIdInput(BaseModel):
    """Arguments accepted by tools that operate on a specific node."""

    node_id: str = Field(min_length=1)


class SearchKnowledgeBaseInput(BaseModel):
    """Arguments accepted by the knowledge-base search tool."""

    query: str = Field(min_length=1)


class DraftRemediationPlanInput(BaseModel):
    """Arguments accepted by the remediation-template lookup tool."""

    action_type: str = Field(min_length=1)
    node_id: str = Field(min_length=1)


class NoArguments(BaseModel):
    """Placeholder schema for tools that take no input arguments."""


class IncidentState(TypedDict, total=False):
    """Mutable shared state carried through the LangGraph workflow."""

    run_id: str
    run_label: str
    source_type: str
    incident: dict[str, Any]
    tool_context: dict[str, Any]
    cross_incident_memory: dict[str, Any]
    profile_summary: str
    profile_source: str
    knowledge_base: dict[str, Any]
    policies: dict[str, Any]
    remediation_templates: dict[str, Any]
    prompts: dict[str, str]
    reasoning_chain: list[str]
    actions_taken: list[dict[str, Any]]
    escalation_history: list[str]
    triage_result: dict[str, Any] | None
    diagnosis_result: dict[str, Any] | None
    remediation_plan: dict[str, Any] | None
    policy_flags: dict[str, Any]
    audit_trail: list[str]
    tool_invocations: list[dict[str, Any]]
    agent_tool_usage: dict[str, list[str]]
    message_log: list[dict[str, Any]]
    stream_audit: bool
    next_route: str
    final_report: str | None


@dataclass
class DataBundle:
    """Container for validated configuration, prompts, and sample inputs."""

    knowledge_base: dict[str, Any]
    policies: dict[str, Any]
    remediation_templates: dict[str, Any]
    alert_prompts: dict[str, Any]
    node_profiles: dict[str, Any]
    prompts: dict[str, str]

    def prompt_by_id(self, prompt_id: str) -> dict[str, Any]:
        """Return one sample alert prompt by identifier."""

        for prompt_entry in self.alert_prompts["prompts"]:
            if prompt_entry["prompt_id"] == prompt_id:
                return deepcopy(prompt_entry)
        available = ", ".join(self.available_prompt_ids())
        raise ValueError(f"Unknown sample alert '{prompt_id}'. Available sample alerts: {available}")

    def available_prompt_ids(self) -> list[str]:
        """List sample alert identifiers in file order for predictable demos."""

        return [prompt_entry["prompt_id"] for prompt_entry in self.alert_prompts["prompts"]]

    def profile_for_host(self, host: str) -> tuple[dict[str, Any], str]:
        """Return a node profile for a host or the configured safe default profile."""

        for profile in self.node_profiles["profiles"]:
            if profile["host"] == host:
                return deepcopy(profile), "host-specific"
        default_profile = deepcopy(self.node_profiles["default_profile"])
        synthesized_profile = {
            "host": host,
            "summary": default_profile["summary"],
            "tool_context": default_profile["tool_context"],
            "cross_incident_memory": default_profile["cross_incident_memory"],
        }
        synthesized_profile["cross_incident_memory"]["node_patterns"].append(
            f"No dedicated node profile exists for {host}; the workflow used the default fallback profile."
        )
        return synthesized_profile, "default"


def load_json_file(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON file and return the parsed payload."""

    return json.loads(path.read_text(encoding="utf-8"))


def load_text_file(path: Path) -> str:
    """Read a UTF-8 text file and strip outer whitespace."""

    return path.read_text(encoding="utf-8").strip()


def validate_model(instance: dict[str, Any], model_cls: type[BaseModel], label: str) -> dict[str, Any]:
    """Validate a payload with Pydantic and return a plain dictionary."""

    try:
        validated = model_cls.model_validate(instance)
    except Exception as exc:
        raise ValueError(f"{label} failed validation: {exc}") from exc
    return validated.model_dump()


def get_arg_value(args: Any, name: str) -> Any:
    """Fetch an argparse attribute without assuming the caller provided it."""

    return getattr(args, name, None)


def load_data_bundle(base_dir: Path, args: argparse.Namespace | Any) -> DataBundle:
    """Load and validate workflow inputs from the project's fixed data folders."""

    data_dir = base_dir / "data"
    prompts_dir = base_dir / "prompts"

    data_files = {
        "knowledge_base": data_dir / "knowledge_base.json",
        "policies": data_dir / "policies.json",
        "remediation_templates": data_dir / "remediation_templates.json",
        "alert_prompts": data_dir / "alert_prompts.json",
        "node_profiles": data_dir / "node_profiles.json",
    }
    prompt_files = {
        "orchestrator": prompts_dir / "orchestrator.md",
        "triage": prompts_dir / "triage_agent.md",
        "diagnosis": prompts_dir / "diagnosis_agent.md",
        "remediation": prompts_dir / "remediation_agent.md",
    }

    payloads = {
        "knowledge_base": validate_model(load_json_file(data_files["knowledge_base"]), KnowledgeBase, "knowledge_base"),
        "policies": validate_model(load_json_file(data_files["policies"]), PolicyConfig, "policies"),
        "remediation_templates": validate_model(
            load_json_file(data_files["remediation_templates"]),
            RemediationTemplateBundle,
            "remediation_templates",
        ),
        "alert_prompts": validate_model(
            load_json_file(data_files["alert_prompts"]),
            AlertPromptBundle,
            "alert_prompts",
        ),
        "node_profiles": validate_model(
            load_json_file(data_files["node_profiles"]),
            NodeProfileBundle,
            "node_profiles",
        ),
    }
    prompts = {name: load_text_file(path) for name, path in prompt_files.items()}

    return DataBundle(
        knowledge_base=payloads["knowledge_base"],
        policies=payloads["policies"],
        remediation_templates=payloads["remediation_templates"],
        alert_prompts=payloads["alert_prompts"],
        node_profiles=payloads["node_profiles"],
        prompts=prompts,
    )


def build_model(args: argparse.Namespace | Any) -> ChatOpenAI:
    """Create the chat model from environment settings, with optional helper overrides."""

    model_name = get_arg_value(args, "model") or os.getenv("OPENAI_MODEL")
    api_key = get_arg_value(args, "api_key") or os.getenv("OPENAI_API_KEY")
    base_url = get_arg_value(args, "base_url") or os.getenv("OPENAI_BASE_URL")
    temperature = get_arg_value(args, "temperature")
    if not model_name:
        raise ValueError("Set OPENAI_MODEL in the environment or .env file before running the workflow.")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY in the environment or .env file before running the workflow.")

    model_kwargs: dict[str, Any] = {}
    if base_url:
        model_kwargs["base_url"] = base_url

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        temperature=0.0 if temperature is None else temperature,
        **model_kwargs,
    )


def ensure_lists(state: IncidentState) -> None:
    """Initialize mutable list and dictionary fields in the workflow state."""

    state.setdefault("reasoning_chain", [])
    state.setdefault("actions_taken", [])
    state.setdefault("escalation_history", [])
    state.setdefault("policy_flags", {})
    state.setdefault("audit_trail", [])
    state.setdefault("tool_invocations", [])
    state.setdefault("agent_tool_usage", {})
    state.setdefault("message_log", [])


def parse_utc_timestamp(timestamp: str) -> datetime:
    """Parse an ISO-8601 timestamp and normalize it to UTC."""

    return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone(timezone.utc)


def serialize_json(data: Any) -> str:
    """Serialize structured data to compact JSON for audit and message logging."""

    return json.dumps(data, default=str, sort_keys=True)


def safe_slug(value: str) -> str:
    """Convert arbitrary text into a filename-safe lowercase slug."""

    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "incident"


def normalize_identifier(value: str) -> str:
    """Convert alert titles into uppercase underscore identifiers."""

    normalized = re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")
    return normalized or "UNKNOWN_ALERT"


def canonical_alert_type(title: str) -> str:
    """Map human alert titles into stable alert identifiers used by the workflow."""

    normalized_title = normalize_identifier(title)
    aliases = {
        "ONE_OR_MORE_CRITICAL_SERVICES_FAILED_ON_COMPUTE_NODES": "CRITICAL_SERVICE_FAILURE",
        "PCIE_BUS_INACCESSIBLE": "PCIE_BUS_INACCESSIBLE",
        "PCIE_LINK_WIDTH_AND_CURRENT_SPEED_MISMATCH": "PCIE_LINK_WIDTH_AND_SPEED_MISMATCH",
        "GPU_UTILIZATION_DROP": "GPU_UTIL_DROP",
    }
    return aliases.get(normalized_title, normalized_title)


def emit_stream_line(state: IncidentState, prefix: str, message: str) -> None:
    """Print live workflow output when audit streaming is enabled."""

    if not state.get("stream_audit"):
        return
    print(f"[{state.get('run_label', 'incident')}] {prefix}{message}\n", flush=True)


def record_message(
    state: IncidentState,
    kind: str,
    actor: str,
    message: str,
    *,
    role: str | None = None,
    timestamp_label: str | None = None,
) -> None:
    """Append a structured message entry to the saved message log."""

    ensure_lists(state)
    state["message_log"].append(
        {
            "kind": kind,
            "actor": actor,
            "role": role,
            "message": message,
            "timestamp": timestamp_label,
        }
    )


def append_audit(state: IncidentState, actor: str, message: str) -> None:
    """Append a timestamped audit entry and stream it to stdout by default."""

    ensure_lists(state)
    timestamp_label = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    formatted = f"[{timestamp_label}] {actor}: {message}"
    state["audit_trail"].append(formatted)
    record_message(state, "audit", actor, message, timestamp_label=timestamp_label)
    emit_stream_line(state, "[AUDIT] ", formatted)


def append_reasoning(state: IncidentState, actor: str, message: str) -> None:
    """Append a reasoning note to shared state and stream it live."""

    ensure_lists(state)
    formatted = f"{actor}: {message}"
    state["reasoning_chain"].append(formatted)
    record_message(state, "reasoning", actor, message)
    emit_stream_line(state, "[REASONING] ", formatted)


def record_llm_message(state: IncidentState, actor: str, role: str, content: str) -> None:
    """Log one prompt, tool-call request, or structured model response."""

    record_message(state, "llm", actor, content, role=role)
    if role in {"system", "user"}:
        return

    role_labels = {
        "assistant": "MESSAGE",
        "tool": "TOOL",
        "tool_result": "TOOL-RESULT",
    }
    label = role_labels.get(role, role.upper())
    emit_stream_line(state, f"[{label}][{actor}] ", content)


def record_tool_invocation(
    state: IncidentState,
    agent_name: str,
    tool_name: str,
    arguments: dict[str, Any],
    result: Any,
) -> None:
    """Store tool I/O for auditability and add an explicit audit entry."""

    ensure_lists(state)
    invocation = {
        "agent": agent_name,
        "tool_name": tool_name,
        "arguments": deepcopy(arguments),
        "result": deepcopy(result),
    }
    state["tool_invocations"].append(invocation)
    used_tools = state["agent_tool_usage"].setdefault(agent_name, [])
    if tool_name not in used_tools:
        used_tools.append(tool_name)
    audit_payload = (
        f"tool {tool_name} invoked with input={serialize_json(arguments)} "
        f"output={serialize_json(result)}"
    )
    append_audit(state, agent_name, audit_payload)
    record_llm_message(
        state,
        agent_name,
        "tool",
        f"{tool_name} input={serialize_json(arguments)} output={serialize_json(result)}",
    )


def lookup_alert_history(state: IncidentState, node_id: str) -> list[dict[str, Any]]:
    """Return recent alerts for the specified node from simulated node context."""

    return deepcopy(state["tool_context"]["alert_history"])


def check_node_status(state: IncidentState, node_id: str) -> dict[str, Any]:
    """Return a simulated node-health snapshot for the specified host."""

    return deepcopy(state["tool_context"]["node_status"])


def get_gpu_error_details(state: IncidentState, node_id: str) -> dict[str, Any]:
    """Return detailed GPU error telemetry for diagnosis."""

    return deepcopy(state["tool_context"]["gpu_error_details"])


def get_running_jobs(state: IncidentState, node_id: str) -> list[dict[str, Any]]:
    """Return simulated workload placement information for the affected node."""

    return deepcopy(state["tool_context"]["running_jobs"])


def search_knowledge_base(state: IncidentState, query: str) -> list[dict[str, Any]]:
    """Search the simulated GPU incident knowledge base using keyword overlap."""

    normalized_query = query.lower()
    matches: list[tuple[int, dict[str, Any]]] = []
    for issue in state["knowledge_base"]["issues"]:
        score = 0
        if issue["issue_id"].lower() in normalized_query:
            score += 3
        for keyword in issue["keywords"]:
            if keyword.lower() in normalized_query:
                score += 1
        if score:
            matches.append((score, deepcopy(issue)))
    matches.sort(key=lambda item: (-item[0], item[1]["issue_id"]))
    return [match for _, match in matches]


def draft_remediation_plan(state: IncidentState, action_type: str, node_id: str) -> dict[str, Any]:
    """Return the configured remediation template for the chosen action type."""

    for template in state["remediation_templates"]["templates"]:
        if template["action_type"] == action_type:
            return deepcopy(template)
    raise ValueError(f"Unknown remediation action_type '{action_type}'.")


def check_blast_radius(state: IncidentState, node_id: str) -> dict[str, Any]:
    """Return the simulated impact of taking the node out of service."""

    return deepcopy(state["tool_context"]["blast_radius"])


def check_maintenance_window(state: IncidentState) -> dict[str, Any]:
    """Return maintenance-window status from the current node profile."""

    return deepcopy(state["tool_context"]["maintenance_window"])


def parse_alert_line(raw_line: str) -> dict[str, Any]:
    """Parse one pipe-delimited alert line into structured fields."""

    line = raw_line.strip()
    if not line:
        raise ValueError("Alert input is empty.")
    if not line.startswith("ALERT:"):
        raise ValueError("Alert input must start with 'ALERT:'")

    alert: dict[str, Any] = {"raw_text": raw_line.strip()}
    parts = [part.strip() for part in line.split(" | ") if part.strip()]
    for index, part in enumerate(parts):
        if index == 0:
            alert["title"] = part.split(":", 1)[1].strip()
            continue
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        normalized_key = key.strip().lower().replace(" ", "_")
        cleaned_value = value.strip()
        if normalized_key == "fire_count":
            alert[normalized_key] = int(cleaned_value)
        else:
            alert[normalized_key] = cleaned_value

    required_fields = ["title", "host", "starts"]
    missing_fields = [field for field in required_fields if field not in alert]
    if missing_fields:
        raise ValueError(f"Alert input is missing required fields: {', '.join(missing_fields)}.")
    alert.setdefault("fire_count", 1)
    return alert


def build_incident_summary(alert: dict[str, Any]) -> str:
    """Create a concise human-readable summary from parsed alert fields."""

    summary_parts = [alert["title"]]
    if alert.get("service"):
        summary_parts.append(f"service={alert['service']}")
    if alert.get("description"):
        summary_parts.append(alert["description"])
    if alert.get("cluster"):
        summary_parts.append(f"cluster={alert['cluster']}")
    return " | ".join(summary_parts)


def build_incident_from_alert(alert: dict[str, Any], source_label: str) -> dict[str, Any]:
    """Normalize a parsed alert into the incident payload used by the workflow."""

    alert_type = canonical_alert_type(alert["title"])
    alert_timestamp = parse_utc_timestamp(alert["starts"])
    alert_id = f"ALT-{alert_timestamp.strftime('%Y%m%d%H%M%S')}-{safe_slug(alert['host']).upper()}"
    services = [alert["service"]] if alert.get("service") else []
    incident = {
        "alert_id": alert_id,
        "node_id": alert["host"],
        "timestamp": alert["starts"],
        "alert_type": alert_type,
        "alert_signal": alert["title"],
        "summary": build_incident_summary(alert),
        "cluster": alert.get("cluster", "unknown-cluster"),
        "oci_name": alert.get("oci_name", "unknown-instance"),
        "serial": alert.get("serial", "unknown-serial"),
        "services": services,
        "related_alerts": [deepcopy(alert)],
        "source_label": source_label,
    }
    return IncidentPayload.model_validate(incident).model_dump()


def build_run_label(source_label: str, alert: dict[str, Any]) -> str:
    """Create a stable label for streaming output and saved artifacts."""

    timestamp = parse_utc_timestamp(alert["starts"]).strftime("%Y%m%dT%H%M%SZ")
    return f"{safe_slug(source_label)}_{safe_slug(alert['host'])}_{timestamp}"


def build_work_item(
    data_bundle: DataBundle,
    *,
    alert_text: str,
    source_label: str,
) -> dict[str, Any]:
    """Create one workflow input object from a single alert line."""

    alert = parse_alert_line(alert_text)
    incident = build_incident_from_alert(alert, source_label)
    profile, profile_source = data_bundle.profile_for_host(incident["node_id"])
    work_item = {
        "run_id": build_run_label(source_label, alert),
        "run_label": source_label,
        "source_type": source_label,
        "incident": incident,
        "tool_context": deepcopy(profile["tool_context"]),
        "cross_incident_memory": deepcopy(profile["cross_incident_memory"]),
        "profile_summary": profile["summary"],
        "profile_source": profile_source,
    }
    if profile_source == "default":
        work_item["cross_incident_memory"]["node_patterns"].append(
            "Fallback node profile used because the host was not defined in the sample profile file."
        )
    return work_item


def work_items_from_sample_alerts(data_bundle: DataBundle, prompt_ids: list[str]) -> list[dict[str, Any]]:
    """Create workflow inputs from sample alerts stored in JSON."""

    work_items: list[dict[str, Any]] = []
    for prompt_id in prompt_ids:
        prompt_entry = data_bundle.prompt_by_id(prompt_id)
        work_item = build_work_item(data_bundle, alert_text=prompt_entry["prompt_text"], source_label=prompt_id)
        work_item["expected_behavior"] = prompt_entry.get("expected_behavior")
        work_item["sample_name"] = prompt_entry["name"]
        work_items.append(work_item)
    return work_items


def is_peak_hours(incident_timestamp: str, policies: dict[str, Any]) -> bool:
    """Return whether the incident falls within configured business hours."""

    local_time = parse_utc_timestamp(incident_timestamp).astimezone(ZoneInfo(policies["peak_hours"]["timezone"]))
    if local_time.weekday() not in policies["peak_hours"]["business_days"]:
        return False
    return policies["peak_hours"]["start_hour"] <= local_time.hour < policies["peak_hours"]["end_hour"]


def disruptive_action_rate_limited(state: IncidentState) -> bool:
    """Return whether recent disruptive actions exceed the configured limit."""

    policies = state["policies"]
    incident_time = parse_utc_timestamp(state["incident"]["timestamp"])
    recent_actions = state["cross_incident_memory"].get("recent_actions", [])
    relevant_action_type = policies["rate_limit"]["action_type"]
    window_minutes = policies["rate_limit"]["window_minutes"]
    max_actions = policies["rate_limit"]["max_actions"]

    recent_count = 0
    for action in recent_actions:
        if action["action_type"] != relevant_action_type:
            continue
        action_time = parse_utc_timestamp(action["timestamp"])
        if incident_time - action_time <= timedelta(minutes=window_minutes):
            recent_count += 1
    return recent_count >= max_actions


def summarize_policy_context(state: IncidentState) -> dict[str, Any]:
    """Return the policy state most relevant to agent decisions."""

    return {
        "severity_gate": state["policies"]["severity_gate"],
        "blast_radius": state["policies"]["blast_radius"],
        "peak_hours": {
            **state["policies"]["peak_hours"],
            "currently_in_peak_hours": is_peak_hours(state["incident"]["timestamp"], state["policies"]),
        },
        "rate_limit": {
            **state["policies"]["rate_limit"],
            "rate_limit_triggered": disruptive_action_rate_limited(state),
        },
        "confidence_threshold": state["policies"]["confidence_threshold"],
    }


def invoke_prompt(
    state: IncidentState,
    model: ChatOpenAI,
    agent_name: str,
    prompt_text: str,
    schema: type[BaseModel],
    payload: dict[str, Any],
) -> BaseModel:
    """Invoke a structured-output prompt and log the exchanged messages."""

    record_llm_message(state, agent_name, "system", prompt_text)
    record_llm_message(state, agent_name, "user", serialize_json(payload))
    messages = [
        SystemMessage(content=prompt_text),
        HumanMessage(content=f"Workflow context as JSON:\n{json.dumps(payload, indent=2)}"),
    ]
    decision = model.with_structured_output(schema).invoke(messages)
    record_llm_message(state, agent_name, "assistant", serialize_json(decision.model_dump()))
    return decision


def invoke_agent_with_tools(
    state: IncidentState,
    model: ChatOpenAI,
    agent_name: str,
    prompt_text: str,
    schema: type[BaseModel],
    payload: dict[str, Any],
    tools: list[BaseTool],
    required_tools: list[str],
) -> BaseModel:
    """Run a tool-using agent loop until required tools are called and a structured answer is produced."""

    tool_map = {tool_instance.name: tool_instance for tool_instance in tools}
    system_prompt = (
        f"{prompt_text}\n\n"
        "Use the provided tools to gather evidence before deciding. "
        f"You must call these tools at least once before finishing: {', '.join(required_tools)}."
    )
    record_llm_message(state, agent_name, "system", system_prompt)
    record_llm_message(state, agent_name, "user", serialize_json(payload))

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Workflow context as JSON:\n{json.dumps(payload, indent=2)}"),
    ]
    bound_model = model.bind_tools(tools)
    used_tools: set[str] = set()

    for _ in range(8):
        response = bound_model.invoke(messages)
        response_log = {
            "content": getattr(response, "content", ""),
            "tool_calls": getattr(response, "tool_calls", []),
        }
        record_llm_message(state, agent_name, "assistant", serialize_json(response_log))
        messages.append(response)

        if getattr(response, "tool_calls", None):
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call.get("args", {})
                tool_instance = tool_map[tool_name]
                result = tool_instance.invoke(tool_args)
                used_tools.add(tool_name)
                tool_message = ToolMessage(
                    content=json.dumps(result, indent=2, default=str),
                    tool_call_id=tool_call["id"],
                )
                messages.append(tool_message)
                record_llm_message(
                    state,
                    agent_name,
                    "tool_result",
                    f"{tool_name} -> {serialize_json(result)}",
                )
            continue

        missing_tools = [tool_name for tool_name in required_tools if tool_name not in used_tools]
        if missing_tools:
            reminder = (
                "You have not finished tool use. "
                f"Call these missing tools before the final decision: {', '.join(missing_tools)}."
            )
            messages.append(HumanMessage(content=reminder))
            record_llm_message(state, agent_name, "user", reminder)
            continue
        break

    missing_tools = [tool_name for tool_name in required_tools if tool_name not in used_tools]
    if missing_tools:
        raise ValueError(f"{agent_name} did not invoke required tools: {', '.join(missing_tools)}.")

    final_instruction = (
        "Using only the workflow context and tool outputs above, return the final structured decision now. "
        "Do not call more tools."
    )
    record_llm_message(state, agent_name, "user", final_instruction)
    final_messages = messages + [HumanMessage(content=final_instruction)]
    decision = model.with_structured_output(schema).invoke(final_messages)
    record_llm_message(state, agent_name, "assistant", serialize_json(decision.model_dump()))
    return decision


def summarize_tool_usage(state: IncidentState, agent_name: str) -> list[str]:
    """Return the distinct tools used by the specified agent."""

    return deepcopy(state.get("agent_tool_usage", {}).get(agent_name, []))


def summarize_state_for_orchestrator(state: IncidentState) -> dict[str, Any]:
    """Return a compact state summary used by the orchestrator prompt."""

    return {
        "incident": state["incident"],
        "triage_result": state.get("triage_result"),
        "diagnosis_result": state.get("diagnosis_result"),
        "remediation_plan": state.get("remediation_plan"),
        "policy_flags": state.get("policy_flags", {}),
        "audit_trail_tail": state.get("audit_trail", [])[-5:],
    }


def enforce_orchestrator_route(state: IncidentState, proposed_route: str) -> str:
    """Apply minimal route enforcement while preserving prompt-driven orchestration."""

    triage = state.get("triage_result")
    diagnosis = state.get("diagnosis_result")
    remediation = state.get("remediation_plan")
    flags = state.get("policy_flags", {})
    diagnosis_required_for = set(state["policies"]["severity_gate"]["diagnosis_required_for"])

    if triage is None:
        return "triage"
    if remediation is not None:
        return "finalize"
    if flags.get("finalize_after_triage"):
        return "finalize"
    if diagnosis is None and triage["severity"] in diagnosis_required_for and triage["escalate"]:
        return proposed_route if proposed_route in {"triage", "diagnosis"} else "diagnosis"
    if diagnosis is not None and remediation is None:
        return proposed_route if proposed_route in {"triage", "remediation"} else "remediation"
    if proposed_route not in {"triage", "diagnosis", "remediation", "finalize"}:
        return "finalize"
    return proposed_route


class IncidentWorkflow:
    """Prompt-driven LangGraph workflow for one GPU incident."""

    def __init__(self, model: ChatOpenAI, data_bundle: DataBundle, stream_audit: bool = True) -> None:
        """Store shared configuration and compile the LangGraph workflow."""

        self.model = model
        self.data_bundle = data_bundle
        self.stream_audit = stream_audit
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build and compile the orchestrator-led multi-agent graph."""

        workflow = StateGraph(IncidentState)
        workflow.add_node("orchestrator", self.orchestrator_node)
        workflow.add_node("triage", self.triage_node)
        workflow.add_node("diagnosis", self.diagnosis_node)
        workflow.add_node("remediation", self.remediation_node)
        workflow.add_node("finalize", self.finalize_node)
        workflow.add_edge(START, "orchestrator")
        workflow.add_conditional_edges(
            "orchestrator",
            lambda state: state["next_route"],
            {
                "triage": "triage",
                "diagnosis": "diagnosis",
                "remediation": "remediation",
                "finalize": "finalize",
            },
        )
        workflow.add_edge("triage", "orchestrator")
        workflow.add_edge("diagnosis", "orchestrator")
        workflow.add_edge("remediation", "orchestrator")
        workflow.add_edge("finalize", END)
        return workflow.compile()

    def new_state(self, work_item: dict[str, Any]) -> IncidentState:
        """Create a fresh incident state object from one normalized input."""

        state: IncidentState = {
            "run_id": work_item["run_id"],
            "run_label": work_item["run_label"],
            "source_type": work_item["source_type"],
            "incident": deepcopy(work_item["incident"]),
            "tool_context": deepcopy(work_item["tool_context"]),
            "cross_incident_memory": deepcopy(work_item["cross_incident_memory"]),
            "profile_summary": work_item["profile_summary"],
            "profile_source": work_item["profile_source"],
            "knowledge_base": deepcopy(self.data_bundle.knowledge_base),
            "policies": deepcopy(self.data_bundle.policies),
            "remediation_templates": deepcopy(self.data_bundle.remediation_templates),
            "prompts": deepcopy(self.data_bundle.prompts),
            "triage_result": None,
            "diagnosis_result": None,
            "remediation_plan": None,
            "final_report": None,
            "stream_audit": self.stream_audit,
        }
        ensure_lists(state)
        append_audit(
            state,
            "Alert received",
            f"{state['incident']['alert_type']} on {state['incident']['node_id']}",
        )
        append_reasoning(state, "Alert received", state["incident"]["summary"])
        append_reasoning(
            state,
            "Profile context",
            f"{state['profile_summary']} (profile source: {state['profile_source']})",
        )
        return state

    def run(self, work_item: dict[str, Any]) -> IncidentState:
        """Run one normalized incident through the full LangGraph workflow."""

        initial_state = self.new_state(work_item)
        return self.graph.invoke(initial_state)

    def build_triage_tools(self, state: IncidentState) -> list[BaseTool]:
        """Create state-bound LangChain tools for the triage agent."""

        agent_name = "Triage Agent"

        @tool("lookup_alert_history", args_schema=NodeIdInput)
        def lookup_alert_history_tool(node_id: str) -> list[dict[str, Any]]:
            """Return recent alerts for the node so triage can detect recurrence patterns."""

            result = lookup_alert_history(state, node_id)
            record_tool_invocation(state, agent_name, "lookup_alert_history", {"node_id": node_id}, result)
            return result

        @tool("check_node_status", args_schema=NodeIdInput)
        def check_node_status_tool(node_id: str) -> dict[str, Any]:
            """Return node-health telemetry such as GPU utilization, temperatures, and uptime."""

            result = check_node_status(state, node_id)
            record_tool_invocation(state, agent_name, "check_node_status", {"node_id": node_id}, result)
            return result

        return [lookup_alert_history_tool, check_node_status_tool]

    def build_diagnosis_tools(self, state: IncidentState) -> list[BaseTool]:
        """Create state-bound LangChain tools for the diagnosis agent."""

        agent_name = "Diagnosis Agent"

        @tool("get_gpu_error_details", args_schema=NodeIdInput)
        def get_gpu_error_details_tool(node_id: str) -> dict[str, Any]:
            """Return XID, ECC, retired-page, and thermal details for deeper diagnosis."""

            result = get_gpu_error_details(state, node_id)
            record_tool_invocation(state, agent_name, "get_gpu_error_details", {"node_id": node_id}, result)
            return result

        @tool("get_running_jobs", args_schema=NodeIdInput)
        def get_running_jobs_tool(node_id: str) -> list[dict[str, Any]]:
            """Return the workloads currently using the node so diagnosis can separate workload from infrastructure faults."""

            result = get_running_jobs(state, node_id)
            record_tool_invocation(state, agent_name, "get_running_jobs", {"node_id": node_id}, result)
            return result

        @tool("search_knowledge_base", args_schema=SearchKnowledgeBaseInput)
        def search_knowledge_base_tool(query: str) -> list[dict[str, Any]]:
            """Return the best matching knowledge-base entries for the supplied query string."""

            result = search_knowledge_base(state, query)
            record_tool_invocation(state, agent_name, "search_knowledge_base", {"query": query}, result)
            return result

        return [get_gpu_error_details_tool, get_running_jobs_tool, search_knowledge_base_tool]

    def build_remediation_tools(self, state: IncidentState) -> list[BaseTool]:
        """Create state-bound LangChain tools for the remediation agent."""

        agent_name = "Remediation Agent"

        @tool("draft_remediation_plan", args_schema=DraftRemediationPlanInput)
        def draft_remediation_plan_tool(action_type: str, node_id: str) -> dict[str, Any]:
            """Return the structured remediation template for the requested action type."""

            result = draft_remediation_plan(state, action_type, node_id)
            record_tool_invocation(
                state,
                agent_name,
                "draft_remediation_plan",
                {"action_type": action_type, "node_id": node_id},
                result,
            )
            return result

        @tool("check_blast_radius", args_schema=NodeIdInput)
        def check_blast_radius_tool(node_id: str) -> dict[str, Any]:
            """Return projected job and user impact for disruptive action on the node."""

            result = check_blast_radius(state, node_id)
            record_tool_invocation(state, agent_name, "check_blast_radius", {"node_id": node_id}, result)
            return result

        @tool("check_maintenance_window", args_schema=NoArguments)
        def check_maintenance_window_tool() -> dict[str, Any]:
            """Return whether the current time falls inside an approved maintenance window."""

            result = check_maintenance_window(state)
            record_tool_invocation(state, agent_name, "check_maintenance_window", {}, result)
            return result

        return [draft_remediation_plan_tool, check_blast_radius_tool, check_maintenance_window_tool]

    def orchestrator_node(self, state: IncidentState) -> IncidentState:
        """Use the prompt-driven orchestrator to choose the next graph route."""

        payload = summarize_state_for_orchestrator(state)
        decision = invoke_prompt(
            state,
            self.model,
            "Orchestrator Agent",
            state["prompts"]["orchestrator"],
            OrchestratorDecision,
            payload,
        )
        next_route = enforce_orchestrator_route(state, decision.route)
        if next_route != decision.route:
            append_audit(
                state,
                "Orchestrator Agent",
                f"{decision.audit_entry} Route adjusted to {next_route} by policy/state enforcement.",
            )
            append_reasoning(
                state,
                "Orchestrator Agent",
                f"Requested route {decision.route} changed to {next_route} because shared state required it.",
            )
        else:
            append_audit(state, "Orchestrator Agent", decision.audit_entry)
            append_reasoning(state, "Orchestrator Agent", decision.reason)
        state["next_route"] = next_route
        return state

    def triage_node(self, state: IncidentState) -> IncidentState:
        """Classify the incident severity and determine whether to escalate."""

        node_id = state["incident"]["node_id"]
        payload = {
            "incident": state["incident"],
            "related_alerts": state["incident"].get("related_alerts", []),
            "node_id": node_id,
            "cross_incident_memory": state["cross_incident_memory"],
            "policy_summary": summarize_policy_context(state),
        }
        decision = invoke_agent_with_tools(
            state,
            self.model,
            "Triage Agent",
            state["prompts"]["triage"],
            TriageDecision,
            payload,
            self.build_triage_tools(state),
            ["lookup_alert_history", "check_node_status"],
        )
        state["triage_result"] = decision.model_dump()
        severity_gate = state["policies"]["severity_gate"]
        finalize_after_triage = (
            decision.auto_resolve
            or not decision.escalate
            or decision.severity in severity_gate["auto_resolve_for"]
        )
        state["policy_flags"]["finalize_after_triage"] = finalize_after_triage
        append_audit(
            state,
            "Triage Agent",
            f"classified severity={decision.severity} category={decision.category} escalating={decision.escalate}",
        )
        append_reasoning(state, "Triage Agent", decision.reasoning)
        for evidence in decision.supporting_evidence:
            append_reasoning(state, "Triage Agent", evidence)
        append_reasoning(
            state,
            "Triage Agent",
            "Tools used: " + ", ".join(summarize_tool_usage(state, "Triage Agent")),
        )
        if finalize_after_triage:
            append_audit(state, "Policy Engine", "Severity gate auto-resolved the incident after triage.")
            state["actions_taken"].append(
                {
                    "action_type": "AUTO_RESOLVE",
                    "reason": "Severity gate or triage outcome blocked deeper automation.",
                }
            )
        return state

    def diagnosis_node(self, state: IncidentState) -> IncidentState:
        """Investigate likely root cause using telemetry, jobs, and the knowledge base."""

        node_id = state["incident"]["node_id"]
        query = " ".join(
            [
                state["incident"]["alert_type"],
                state["incident"]["alert_signal"],
                state["triage_result"]["category"],
                " ".join(state["cross_incident_memory"].get("node_patterns", [])),
            ]
        )
        payload = {
            "incident": state["incident"],
            "triage_result": state["triage_result"],
            "node_id": node_id,
            "knowledge_base_query": query,
            "related_alerts": state["incident"].get("related_alerts", []),
            "cross_incident_memory": state["cross_incident_memory"],
        }
        decision = invoke_agent_with_tools(
            state,
            self.model,
            "Diagnosis Agent",
            state["prompts"]["diagnosis"],
            DiagnosisDecision,
            payload,
            self.build_diagnosis_tools(state),
            ["get_gpu_error_details", "get_running_jobs", "search_knowledge_base"],
        )
        state["diagnosis_result"] = decision.model_dump()
        low_confidence = decision.confidence < state["policies"]["confidence_threshold"]
        state["policy_flags"]["manual_investigation_required"] = low_confidence
        append_audit(
            state,
            "Diagnosis Agent",
            f"root cause={decision.root_cause} confidence={decision.confidence:.2f}",
        )
        append_reasoning(state, "Diagnosis Agent", decision.reasoning)
        for evidence in decision.evidence:
            append_reasoning(state, "Diagnosis Agent", evidence)
        append_reasoning(
            state,
            "Diagnosis Agent",
            "Knowledge-base matches: " + ", ".join(decision.knowledge_base_matches or ["None"]),
        )
        append_reasoning(
            state,
            "Diagnosis Agent",
            "Tools used: " + ", ".join(summarize_tool_usage(state, "Diagnosis Agent")),
        )
        if low_confidence:
            append_audit(
                state,
                "Policy Engine",
                "Diagnosis confidence fell below threshold. Manual investigation will be required.",
            )
        return state

    def remediation_node(self, state: IncidentState) -> IncidentState:
        """Recommend a remediation plan, then enforce safety policies on the result."""

        node_id = state["incident"]["node_id"]
        payload = {
            "incident": state["incident"],
            "triage_result": state["triage_result"],
            "diagnosis_result": state["diagnosis_result"],
            "node_id": node_id,
            "related_alerts": state["incident"].get("related_alerts", []),
            "available_action_types": [
                {
                    "action_type": template["action_type"],
                    "action": template["action"],
                    "disruptive": template["disruptive"],
                }
                for template in state["remediation_templates"]["templates"]
            ],
            "policy_summary": summarize_policy_context(state),
            "recent_actions": state["cross_incident_memory"].get("recent_actions", []),
        }
        decision = invoke_agent_with_tools(
            state,
            self.model,
            "Remediation Agent",
            state["prompts"]["remediation"],
            RemediationDecision,
            payload,
            self.build_remediation_tools(state),
            ["draft_remediation_plan", "check_blast_radius", "check_maintenance_window"],
        )

        blast_radius_results = [
            invocation["result"]
            for invocation in state.get("tool_invocations", [])
            if invocation["agent"] == "Remediation Agent" and invocation["tool_name"] == "check_blast_radius"
        ]
        maintenance_window_results = [
            invocation["result"]
            for invocation in state.get("tool_invocations", [])
            if invocation["agent"] == "Remediation Agent" and invocation["tool_name"] == "check_maintenance_window"
        ]
        blast_radius = deepcopy(blast_radius_results[-1])
        maintenance_window = deepcopy(maintenance_window_results[-1])
        peak_hours = is_peak_hours(state["incident"]["timestamp"], state["policies"])
        rate_limited = disruptive_action_rate_limited(state)
        blast_radius_threshold = state["policies"]["blast_radius"]["human_approval_job_threshold"]
        low_confidence = state["policy_flags"].get("manual_investigation_required", False)

        enforced_action_type = decision.action_type
        approval = decision.approval
        safeguards_triggered: list[str] = []

        if low_confidence:
            enforced_action_type = "MANUAL_INVESTIGATION"
            approval = "HUMAN-REQUIRED"
            safeguards_triggered.append("Confidence threshold triggered manual investigation.")

        template = draft_remediation_plan(state, enforced_action_type, node_id)
        disruptive = template["disruptive"]

        if disruptive and peak_hours and not maintenance_window["in_window"]:
            if enforced_action_type == "TAG_AND_TERMINATE":
                approval = "HUMAN-REQUIRED"
                safeguards_triggered.append(
                    "Peak-hours protection requires human approval before tag-and-terminate can proceed."
                )
            else:
                enforced_action_type = "SCHEDULE_MAINTENANCE"
                approval = "HUMAN-REQUIRED"
                safeguards_triggered.append(
                    "Peak-hours protection blocked an immediate drain and forced scheduled maintenance."
                )
                template = draft_remediation_plan(state, enforced_action_type, node_id)
                disruptive = template["disruptive"]

        if disruptive and blast_radius["affected_job_count"] > blast_radius_threshold:
            approval = "HUMAN-REQUIRED"
            safeguards_triggered.append("Blast-radius threshold exceeded. Human approval is required.")
            escalation_message = (
                f"ESCALATION: Human approval required because action would affect "
                f"{blast_radius['affected_job_count']} jobs."
            )
            state["escalation_history"].append(escalation_message)
            append_audit(state, "Remediation Agent", escalation_message)

        if disruptive and rate_limited:
            enforced_action_type = "MANUAL_INVESTIGATION"
            approval = "HUMAN-REQUIRED"
            safeguards_triggered.append("Rate limit blocked additional disruptive action in the current hour.")
            template = draft_remediation_plan(state, enforced_action_type, node_id)

        if not template.get("rollback_plan"):
            raise ValueError("Every remediation plan must include a rollback plan.")

        state["remediation_plan"] = {
            "action_type": enforced_action_type,
            "recommended_action": template["action"],
            "steps": template["steps"],
            "estimated_resolution": template["estimated_resolution"],
            "rollback_plan": template["rollback_plan"],
            "approval": approval,
            "approval_required": approval == "HUMAN-REQUIRED",
            "justification": decision.justification,
            "blast_radius": blast_radius,
            "blast_radius_summary": decision.blast_radius_summary,
            "maintenance_window": maintenance_window,
            "reasoning": decision.reasoning,
            "safeguards_triggered": safeguards_triggered,
        }
        state["actions_taken"].append({"action_type": enforced_action_type, "approval": approval})
        append_audit(
            state,
            "Remediation Agent",
            f"recommended {template['action']} with approval={approval}",
        )
        append_reasoning(state, "Remediation Agent", decision.reasoning)
        append_reasoning(
            state,
            "Remediation Agent",
            "Tools used: " + ", ".join(summarize_tool_usage(state, "Remediation Agent")),
        )
        for safeguard in safeguards_triggered:
            append_reasoning(state, "Policy Engine", safeguard)
        return state

    def finalize_node(self, state: IncidentState) -> IncidentState:
        """Build the final human-readable incident report."""

        state["final_report"] = build_incident_report(state)
        return state


def format_list_items(items: list[str], indent: str = "    - ") -> list[str]:
    """Format a list of strings into report-ready bullet lines."""

    if not items:
        return [f"{indent}None"]
    return [f"{indent}{item}" for item in items]


def build_architecture_mermaid(base_dir: Path) -> str:
    """Build the workflow graph and return LangGraph's Mermaid source."""

    diagram_args = build_runtime_args_namespace(model="diagram-only", api_key="diagram-only", stream_audit=False)
    _, workflow = create_workflow(base_dir, diagram_args)
    return workflow.graph.get_graph().draw_mermaid()


def export_architecture_mermaid(base_dir: Path, output_path: Path) -> Path:
    """Write the workflow architecture Mermaid source to disk and return its path."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_architecture_mermaid(base_dir), encoding="utf-8")
    return output_path


def ensure_architecture_artifact(base_dir: Path, output_dir: Path) -> Path:
    """Generate the Mermaid workflow source in `outputs/` for reproducible report artifacts."""

    return export_architecture_mermaid(base_dir, output_dir / "langgraph_workflow.mmd")


def build_incident_report(state: IncidentState) -> str:
    """Render a readable incident report from the final workflow state."""

    incident = state["incident"]
    triage = state.get("triage_result")
    diagnosis = state.get("diagnosis_result")
    remediation = state.get("remediation_plan")
    timestamp_utc = parse_utc_timestamp(incident["timestamp"]).strftime("%Y-%m-%d %H:%M UTC")
    divider = "═" * 59

    lines = [
        divider,
        f"INCIDENT REPORT: {incident['alert_id']}",
        f"Node: {incident['node_id']} | Alert: {incident['alert_type']} | Time: {timestamp_utc}",
        divider,
        "",
        "TRIAGE (Agent 1)",
    ]
    if triage:
        lines.extend(
            [
                f"  Severity: {triage['severity']}",
                f"  Category: {triage['category'].title()}",
                f"  Escalate: {'Yes -> Diagnosis Agent' if triage['escalate'] else 'No -> Auto-resolve'}",
                f"  Tools Used: {', '.join(summarize_tool_usage(state, 'Triage Agent')) or 'None'}",
                f"  Reasoning: \"{triage['reasoning']}\"",
            ]
        )
    else:
        lines.append("  Not executed.")

    related_alerts = incident.get("related_alerts", [])
    if related_alerts:
        lines.extend(["  Source Alert:", f"    {related_alerts[0]['raw_text']}"])

    lines.extend(["", "DIAGNOSIS (Agent 2)"])
    if diagnosis:
        lines.extend(
            [
                f"  Root Cause: {diagnosis['root_cause']} (confidence: {diagnosis['confidence']:.2f})",
                f"  Tools Used: {', '.join(summarize_tool_usage(state, 'Diagnosis Agent')) or 'None'}",
                "  Evidence:",
                *format_list_items(diagnosis["evidence"]),
                "  Knowledge Base Match:",
                *format_list_items(diagnosis["knowledge_base_matches"]),
            ]
        )
    else:
        lines.append("  Skipped by the severity gate or triage auto-resolution.")

    lines.extend(["", "REMEDIATION (Agent 3)"])
    if remediation:
        blast_radius = remediation["blast_radius"]
        lines.extend(
            [
                f"  Recommended Action: {remediation['recommended_action']}",
                f"  Tools Used: {', '.join(summarize_tool_usage(state, 'Remediation Agent')) or 'None'}",
                "  Steps:",
            ]
        )
        lines.extend([f"    {index}. {step}" for index, step in enumerate(remediation["steps"], start=1)])
        lines.extend(
            [
                f"  Blast Radius: {blast_radius['affected_job_count']} jobs, {blast_radius['affected_user_count']} users affected",
                f"  Approval: {remediation['approval']}",
                f"  Rollback: {remediation['rollback_plan']}",
                f"  Estimated Resolution: {remediation['estimated_resolution']}",
            ]
        )
        if remediation["safeguards_triggered"]:
            lines.extend(["  Safeguards Triggered:", *format_list_items(remediation["safeguards_triggered"])])
    else:
        lines.append("  Not required because the incident auto-resolved after triage.")

    lines.extend(["", "AUDIT TRAIL"])
    lines.extend([f"  {entry}" for entry in state.get("audit_trail", [])])
    return "\n".join(lines)


def save_outputs(state: IncidentState, output_dir: Path) -> None:
    """Persist the final report and full state, including messages and tool I/O."""

    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = state["run_id"]
    (output_dir / f"{run_id}_report.txt").write_text(state["final_report"] or "", encoding="utf-8")
    (output_dir / f"{run_id}_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def execute_workflow_runs(
    workflow: IncidentWorkflow,
    work_items: list[dict[str, Any]],
    output_dir: Path,
    *,
    show_progress: bool,
) -> list[IncidentState]:
    """Run one or more normalized workflow inputs and save each result."""

    final_states: list[IncidentState] = []
    total = len(work_items)
    for index, work_item in enumerate(work_items, start=1):
        if show_progress:
            print(f"==> Running alert {index}/{total}: {work_item['run_label']}", flush=True)
        final_state = workflow.run(work_item)
        final_states.append(final_state)
        save_outputs(final_state, output_dir)
        if show_progress:
            remediation = final_state.get("remediation_plan")
            outcome = remediation["recommended_action"] if remediation else "AUTO-RESOLVED AFTER TRIAGE"
            print(f"<== Completed alert {index}/{total}: {work_item['run_label']} -> {outcome}", flush=True)
    return final_states


def create_workflow(base_dir: Path, args: Any) -> tuple[DataBundle, IncidentWorkflow]:
    """Load validated data and construct a ready-to-run workflow object."""

    data_bundle = load_data_bundle(base_dir, args)
    workflow = IncidentWorkflow(build_model(args), data_bundle, stream_audit=bool(get_arg_value(args, "stream_audit")))
    return data_bundle, workflow


def resolve_work_items(data_bundle: DataBundle, args: Any) -> list[dict[str, Any]]:
    """Resolve the requested execution mode into normalized workflow inputs."""

    if args.sample_alert:
        return work_items_from_sample_alerts(data_bundle, [args.sample_alert])
    raise ValueError("No sample alert was selected.")


def run_workflow_request(
    *,
    base_dir: Path,
    args: Any,
    sample_alert: str | None = None,
    alert_text: str | None = None,
    source_label: str = "interactive",
    output_dir: Path | None = None,
) -> list[IncidentState]:
    """Programmatic entry point for notebook cells and other local executions."""

    ensure_architecture_artifact(base_dir, output_dir or base_dir / "outputs")
    data_bundle, workflow = create_workflow(base_dir, args)
    if sample_alert and alert_text:
        raise ValueError("Provide either sample_alert or alert_text, not both.")
    if sample_alert:
        work_items = resolve_work_items(data_bundle, argparse.Namespace(sample_alert=sample_alert))
    elif alert_text:
        work_items = [build_work_item(data_bundle, alert_text=alert_text, source_label=source_label)]
    else:
        raise ValueError("Provide sample_alert or alert_text.")
    return execute_workflow_runs(
        workflow,
        work_items,
        output_dir or base_dir / "outputs",
        show_progress=bool(get_arg_value(args, "stream_audit")),
    )


def build_runtime_args_namespace(
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.0,
    stream_audit: bool = True,
) -> argparse.Namespace:
    """Create an argparse-like namespace for notebook or library-style execution."""

    return argparse.Namespace(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        stream_audit=stream_audit,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the small academic-project CLI surface."""

    parser = argparse.ArgumentParser(
        description="Prompt-driven multi-agent GPU cluster incident-response workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python agentic_system.py --sample-alert xid_48_hardware_failure\n"
            "  python agentic_system.py --interactive\n"
            "  python agentic_system.py --validate-only"
        ),
    )
    mode_group = parser.add_argument_group("workflow mode")
    execution_mode = mode_group.add_mutually_exclusive_group(required=True)
    execution_mode.add_argument("--sample-alert", help="Run one sample alert from alert_prompts.json.")
    execution_mode.add_argument("--interactive", action="store_true", help="Start the interactive single-alert loop.")
    execution_mode.add_argument("--validate-only", action="store_true", help="Validate prompts and data files, then exit.")

    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        parser.print_help(sys.stderr)
        parser.exit(2, "\nerror: one workflow mode is required\n")
    return parser.parse_args(argv)


def list_sample_alerts(data_bundle: DataBundle) -> None:
    """Print sample alert identifiers and short descriptions for easy discovery."""

    for prompt_entry in data_bundle.alert_prompts["prompts"]:
        suffix = f" | expected: {prompt_entry['expected_behavior']}" if prompt_entry.get("expected_behavior") else ""
        print(f"{prompt_entry['prompt_id']}: {prompt_entry['name']}{suffix}")


def print_final_reports(final_states: list[IncidentState]) -> None:
    """Print final incident reports for one or more completed workflow runs."""

    for index, final_state in enumerate(final_states):
        print(final_state["final_report"])
        if index < len(final_states) - 1:
            print()


def interactive_alert_loop(base_dir: Path, args: argparse.Namespace, data_bundle: DataBundle) -> None:
    """Start an interactive loop that runs the full workflow for each entered alert line."""

    workflow = IncidentWorkflow(build_model(args), data_bundle, stream_audit=args.stream_audit)
    output_dir = base_dir / "outputs"
    print("Interactive alert loop started. Enter one alert per line. Type 'quit' or 'exit' to stop.")
    print("Each alert triggers the full orchestrator -> triage -> diagnosis -> remediation workflow.")
    print(f"Reports and state snapshots are saved automatically to {output_dir}.")
    while True:
        try:
            raw_input_line = input("alert> ").strip()
        except EOFError:
            print()
            break
        if raw_input_line.lower() in {"quit", "exit"}:
            break
        if not raw_input_line:
            continue
        try:
            work_item = build_work_item(data_bundle, alert_text=raw_input_line, source_label="interactive")
            final_state = workflow.run(work_item)
            save_outputs(final_state, output_dir)
            print(final_state["final_report"])
        except Exception as exc:
            print(f"Input error: {exc}", file=sys.stderr)


def main() -> None:
    """CLI entry point for validation, sample runs, and the interactive alert loop."""

    load_dotenv()
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "outputs"
    args.stream_audit = True
    architecture_path = ensure_architecture_artifact(base_dir, output_dir)
    data_bundle = load_data_bundle(base_dir, args)
    print(f"Architecture Mermaid written to {architecture_path}")

    if args.validate_only:
        print("All prompt files and data files validated successfully.")
        return

    if args.interactive:
        interactive_alert_loop(base_dir, args, data_bundle)
        return

    workflow = IncidentWorkflow(build_model(args), data_bundle, stream_audit=args.stream_audit)
    work_items = resolve_work_items(data_bundle, args)
    final_states = execute_workflow_runs(
        workflow,
        work_items,
        output_dir,
        show_progress=args.stream_audit,
    )
    print_final_reports(final_states)


if __name__ == "__main__":
    main()
