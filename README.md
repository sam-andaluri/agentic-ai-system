# Agentic AI System

## Multi-Agent GPU Cluster Incident Response System

This project implements a **multi-agent GPU cluster incident response system** for academic use. The workflow accepts a single alert, routes it through an orchestrator, lets specialist agents triage the issue, investigate likely root cause, and recommend a safe remediation plan with safeguards, auditability, and rollback requirements.

The system is implemented with **LangGraph**, **LangChain tools**, **Pydantic models**, external JSON data files, and Markdown prompt templates. It is recommendation-only: it does not directly execute infrastructure changes.

## 1. Cloning repo

Clone the repository and move into the project folder.

```bash
git clone https://github.com/sam-andaluri/agentic-ai-system.git
cd agentic-ai-system
```

## 2. Project Files

Main implementation and supporting files:

- `agentic_system.py`
- `models.py`
- `data/`
- `prompts/`
- `outputs/`
- `Agentic_AI_System_Design_Report.md`
- `Agentic_AI_System_Design_Report.pdf`


## 3. Install `uv`

Install `uv` with the standalone installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or install it with Homebrew on macOS:

```bash
brew install uv
```

Confirm the installation:

```bash
uv --version
```

## 4. Prerequisites

Make sure the following tools are available on your system:

- Python 3.10 or higher
- Pandoc for PDF generation

Quick checks:

```bash
python --version
pandoc --version
```

## 5. Create and Activate a Virtual Environment

From the project folder:

```bash
cd agentic-ai-system
uv venv .venv --python 3.13.5
source .venv/bin/activate
```

## 6. Install Dependencies

Install the pinned dependencies from `requirements.txt`:

```bash
uv pip install -r requirements.txt
```

## 7. Setup dot env for model access

Copy the example env file:

```bash
cp .env.example .env
```

Populate `.env` with your model settings:

```dotenv
OPENAI_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_openai_api_key_here
# OPENAI_BASE_URL=https://api.openai.com/v1
```

The Python runtime automatically loads `.env` with `python-dotenv`.

## 8. Validate Project Inputs

Validate the prompt files and JSON data files before running the workflow:

```bash
source .venv/bin/activate
python agentic_system.py --validate-only
```

This checks the external data files in `data/` and the prompt templates in `prompts/`.

## 9. Run the Agentic Workflow

**Alert input format**

The workflow accepts one pipe-delimited alert line per input.

Minimum required fields:
- ALERT: <alert title>
- Host: <node or host name>
- Starts: <ISO-8601 UTC timestamp>

Optional fields:
- Service: <service name>
- Fire Count: <integer>
- Description: <free text>
- Cluster: <cluster name>
- OCI name: <instance name>
- Serial: <serial number>
- Panel URL: <url or N/A>
- Silence URL: <url or N/A>

Required format:
ALERT: <title> | Host: <host> | Starts: <timestamp>

Recommended full format:

ALERT: <title> | Host: <host> | Service: <service> | Starts: <timestamp> | Fire Count: <integer> | Description: <description> | Cluster: <cluster> | OCI name: <oci_name> | Serial: <serial> | Panel URL: <url_or_na> |
Silence URL: <url_or_na>

Example minimum valid prompt:

ALERT: GPU utilization drop | Host: gpu-node-017 | Starts: 2026-04-19T16:05:00Z

Some sample prompts:

Software alert
```
ALERT: One or more critical services failed on compute nodes. | Host: gpu-host-demo-001 | Service: nvml-exporter.service | Starts: 2026-04-21T11:10:00Z | Fire Count: 3 | Description: Check and restart failed critical s
  ervices on compute node. | Cluster: cluster-demo-d | OCI name: inst-demo-svc-001 | Serial: SERIAL-DEMO-SVC-001 | Panel URL: N/A | Silence URL: https://example.invalid/silence/service
```

Hardware alert
```
ALERT: XID 79 GPU has fallen off the bus | Host: gpu-node-005 | Starts: 2026-04-21T18:00:00Z | Fire Count: 1 | Description: Severe connectivity failure; drain may impact active users. | Cluster: cluster-demo-c | OCI na
  me: inst-demo-005 | Serial: SERIAL-DEMO-005 | Panel URL: N/A | Silence URL: https://example.invalid/silence/xid79
```

Run the interactive alert loop:

```bash
source .venv/bin/activate
python agentic_system.py --interactive
```

You can run one sample alert through the Agentic workflow without running the interactive loop by

```bash
python agentic_system.py --sample-alert recurring_xid_63_pattern
```

Current sample alert ids:

- `xid_48_hardware_failure`
- `gpu_utilization_false_positive`
- `recurring_xid_63_pattern`
- `xid_79_peak_hours_safeguard`
- `critical_service_restart`
- `pcie_bus_inaccessible`

Each run prints a live audit trail and also saves artifacts to `outputs/`, including:

- incident report text files
- full state snapshots in JSON
- audit trail entries
- reasoning chain entries
- tool inputs and outputs
- saved message logs

## 10. Generate the Report PDF

If you update the report markdown, regenerate the PDF with:

```bash
pandoc Agentic_AI_System_Design_Report.md -o Agentic_AI_System_Design_Report.pdf
```

## 11. Convert LangChain mermaid file to png

Install mermaid-cli: https://github.com/mermaid-js/mermaid-cli

```bash
mmdc -i outputs/langgraph_workflow.mmd -o outputs/langgraph_workflow.png -b white -s 2
```

## 12. Generate Final `requirements.txt`

After running the notebook or script in your final environment, regenerate the exact package versions:

```bash
uv pip freeze > requirements.txt
```

## References

LangChain. (n.d.). *LangGraph documentation*.

National Institute of Standards and Technology. (2023). *AI risk management framework (AI RMF 1.0)*. U.S. Department of Commerce.

Schick, T., Dwivedi-Yu, J., Dessi, R., Raileanu, R., Lomeli, M., Hambro, E., Zettlemoyer, L., Cancedda, N., & Scialom, T. (2023). *Toolformer: Language models can teach themselves to use tools*.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). *ReAct: Synergizing reasoning and acting in language models*.
