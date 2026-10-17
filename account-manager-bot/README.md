# Account Manager Bot 🤖📋

An intelligent question-answering assistant built with **LangChain** and **LLMs** that answers questions from loaded PDF documents. Designed for account management, sales support, and enterprise knowledge retrieval.

---

## Architecture

```
PDF Documents
    │
    ▼ PyPDFLoader / DirectoryLoader
Raw Pages (LangChain Document objects)
    │
    ▼ RecursiveCharacterTextSplitter
    │  chunk_size=800, chunk_overlap=150
Text Chunks
    │
    ▼ HuggingFaceEmbeddings (local) OR OpenAIEmbeddings
Vector Embeddings
    │
    ▼ ChromaDB (persisted to disk)
Vector Store
    │
    ▼ MMR Retriever (top-K diverse chunks)
Relevant Context
    │
    ▼ ConversationalRetrievalChain
    │  + ConversationBufferWindowMemory (last 6 turns)
    │  + Custom QA Prompt + Condense Question Prompt
LLM (GPT-4o-mini / Claude / local model)
    │
    ▼
Answer + Source Citations
    │
    ▼ Gradio Web Interface
User
```

---

## Setup — Development

### Prerequisites

- Python 3.11+
- Git
- (Optional) OpenAI or Anthropic API key

### Step 1: Clone and set up environment

```bash
git clone https://github.com/your-org/account-manager-bot.git
cd account-manager-bot

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Choose one LLM backend:
OPENAI_API_KEY=sk-your-key-here           # OpenAI
# ANTHROPIC_API_KEY=sk-ant-your-key-here  # OR Anthropic Claude

# Optional LLM settings
OPENAI_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.0
MAX_TOKENS=1024

# Embeddings (local by default — no API key needed)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Document processing
CHUNK_SIZE=800
CHUNK_OVERLAP=150
RETRIEVER_K=5

# Paths
DOCS_PATH=./docs
VECTOR_STORE_PATH=./data/vectorstore

# App
ENVIRONMENT=development
LOG_LEVEL=INFO
GRADIO_SERVER_PORT=7860
GRADIO_SHARE=false

# Security (production)
ENABLE_AUTH=false
GRADIO_AUTH_USER=admin
GRADIO_AUTH_PASS=change-me
```

### Step 3: Add your PDF documents

```bash
mkdir -p docs
cp /path/to/your/company-docs/*.pdf docs/
```

### Step 4: Run the app

```bash
python -m src.app
```

Open http://localhost:7860 in your browser.

### Step 5: Run tests

```bash
pytest tests/ -v --cov=src --cov=config
```

---

## Project Structure

```
account-manager-bot/
├── src/
│   ├── bot.py           Core LangChain RAG pipeline
│   └── app.py           Gradio web interface
├── config/
│   └── settings.py      Pydantic settings (env var loaded)
├── tests/
│   ├── test_bot.py      Unit + integration tests
│   └── conftest.py      Pytest fixtures
├── docs/                PDF documents (gitignored)
├── data/
│   └── vectorstore/     Persisted ChromaDB (gitignored)
├── scripts/
│   ├── ingest.py        CLI batch indexing tool
│   └── healthcheck.py   Deployment health check script
├── helm/                Kubernetes Helm chart
├── .github/workflows/
│   └── deploy.yml       CI/CD pipeline (dev→staging→prod)
├── Dockerfile           Multi-stage production Docker image
├── requirements.txt     Pinned Python dependencies
├── .env.example         Environment variable template
└── README.md            This file
```

---

## Component Deep Dive

### 1. Document Loader (PyPDFLoader)

Loads PDF documents page-by-page, preserving source metadata (filename, page number). Supports single files and entire directories via `DirectoryLoader`.

### 2. Text Splitter (RecursiveCharacterTextSplitter)

Splits on paragraph → sentence → word boundaries. `chunk_size=800` balances context window usage vs. retrieval granularity. `chunk_overlap=150` ensures no context is lost at chunk boundaries.

### 3. Embeddings (HuggingFaceEmbeddings)

`all-MiniLM-L6-v2` runs entirely locally — no API key required. Generates 384-dimensional vectors. For higher accuracy, switch to `text-embedding-3-small` (OpenAI) via the `OPENAI_API_KEY` setting.

### 4. Vector Store (ChromaDB)

Persists embeddings to disk. Uses **MMR (Maximal Marginal Relevance)** retrieval to return diverse, relevant chunks — avoiding duplicate context from the same document section.

### 5. LLM (GPT-4o-mini / Claude)

`temperature=0.0` ensures deterministic, grounded answers. Custom prompt template instructs the model to answer only from context — preventing hallucination from external knowledge.

### 6. Memory (ConversationBufferWindowMemory)

Keeps the last 6 turns of conversation, enabling follow-up questions without re-stating context. Memory is tied to the user session and can be reset via the Clear button.

### 7. Gradio Interface

Production-ready chat UI with file upload, source citations, example questions, and session management.

---

## Pilot → Production Deployment

### Stage 1: Pilot Validation (Weeks 1–4)

**Goal:** Validate accuracy and usability with a small user group (3–5 account managers).

**Steps:**
1. Deploy to dev environment (single Docker container, no auth required)
2. Load a representative sample of company PDFs (10–20 documents)
3. Run shadow evaluation: collect 50+ real questions + expected answers from AMs
4. Measure retrieval accuracy (manual evaluation) and response latency
5. Document all hallucinations or incorrect answers in a validation log

**Acceptance criteria:**
- Answer relevance score ≥ 80% (human evaluation by AMs)
- Response latency p99 ≤ 5 seconds
- Zero PII exposure in source citations
- All questions answered or gracefully declined (no hallucinations)

### Stage 2: Limited Production (Weeks 5–8)

**Steps:**
1. Enable Gradio auth (`ENABLE_AUTH=true`)
2. Deploy to staging Kubernetes cluster (2 replicas, HPA enabled)
3. Connect to company SSO (SAML/OAuth) via reverse proxy
4. Enable audit logging to CloudWatch / Splunk
5. Load all approved company PDFs via automated ingestion pipeline
6. Run integration tests and load test (50 concurrent users)

### Stage 3: Full Production (Weeks 9–12)

**Steps:**
1. CISO sign-off on all governance gates (see section below)
2. Create GitHub Release → triggers production pipeline
3. Two required approvers in GitHub Environments (Eng Lead + CISO rep)
4. Helm rolling update: 3 replicas, zero-downtime deployment
5. CloudWatch alarms: latency, error rate, LLM API failures
6. Update company intranet with user guide and feedback form

### Stage 4: Steady-State Operations

- **Document refresh:** Automated PDF re-ingestion nightly via Airflow/EventBridge
- **Model updates:** Quarterly review of LLM version and embedding model performance
- **Drift detection:** Monitor answer quality via random sampling and user feedback
- **Rollback:** `helm rollback account-manager-bot -n account-bot` for immediate revert

---

## CISO Governance — Data Security Framework

### Data Classification

| Data Element | Classification | Handling Requirement |
|---|---|---|
| PDF source documents | Confidential — Internal | Encrypted at rest (AES-256/KMS); access via IAM role only; never logged in full |
| ChromaDB vector embeddings | Confidential — Internal | Stored in encrypted EBS/EFS; embeddings are mathematical representations, not raw text |
| User questions | Internal — Transient | Hashed (SHA-256) in audit logs; never stored in plaintext |
| LLM API payloads | Confidential — Transient | TLS 1.3 in transit; no retention at LLM provider (Business API tier required) |
| Bot answers | Internal | Stored in session only; audit log records length/timestamp, not content |
| Conversation history | Confidential — Session | In-memory only; cleared on session end; never persisted to disk |

### Access Control

| Role | Documents | Vector Store | App | Admin |
|---|---|---|---|---|
| Account Manager (end user) | None (no direct access) | None | Chat interface only | ❌ |
| ML Engineer | Read/write dev | Read/write dev | Dev/staging | ❌ |
| ML Ops | Read prod | Read prod | All environments | Limited |
| Security Auditor | Read audit logs only | None | None | ❌ |
| Admin (CISO approved) | All (with JIT access) | All (JIT) | All | ✅ (JIT, 4hr max) |

### PII and Data Leakage Prevention

1. **No PII in training data**: PDFs must be reviewed before ingestion to ensure no customer PII (names, SSNs, account numbers) is present in the vector store.
2. **Source citation redaction**: Source excerpts in the UI are truncated to 200 characters and reviewed for PII before display.
3. **Audit log design**: The `_conversation_log` stores only `question_hash`, `source_count`, `answer_length`, and `timestamp` — never raw question or answer text.
4. **LLM API privacy**: Ensure OpenAI API key is a Business/Enterprise account with zero data retention opt-out. For maximum privacy, use a locally-hosted model (Ollama + Llama 3) with no external data transmission.

### Encryption

| Layer | Standard | Key Management |
|---|---|---|
| Docs at rest | AES-256 via AWS KMS | Customer-managed key, 90-day rotation |
| Vector store at rest | AES-256 EBS encryption | AWS-managed KMS |
| LLM API calls in transit | TLS 1.3 | Managed by cloud provider |
| Secrets | AWS Secrets Manager | Auto-rotation enforced |
| Gradio session cookies | HTTPS + HSTS | TLS cert via ACM |

### Network Security

```
Internet
    │
    ▼ CloudFront / WAF (rate limiting, SQL injection protection)
    │
    ▼ ALB (TLS termination, HSTS)
    │
    ▼ EKS Private Subnet (no public IP on pods)
    │
    ├─► Gradio App Pod
    │       │ Outbound ONLY to LLM API (via NAT Gateway + fixed egress IP)
    │
    └─► ChromaDB / EFS (isolated storage subnet)
```

### Dependency Security

- **Trivy** scans container images on every PR (blocks on HIGH/CRITICAL CVEs)
- **Bandit** performs Python SAST scanning in CI
- **Safety** checks all Python dependencies for known vulnerabilities
- **Snyk** monitors for new CVEs in production dependencies continuously
- All Python packages pinned to exact versions in `requirements.txt`

### CISO Sign-Off Checklist (before Production)

```
Data Protection:
  ☐ LLM provider Data Processing Agreement executed (zero retention)
  ☐ PDF documents reviewed for PII before ingestion
  ☐ Vector store encryption at rest verified
  ☐ Audit log design reviewed (no raw question/answer storage)

Access Control:
  ☐ IAM least-privilege role matrix approved
  ☐ Gradio authentication enabled (SSO or username/password)
  ☐ JIT admin access configured (max 4-hour session)
  ☐ All secrets in AWS Secrets Manager (no .env files in production)

Network Security:
  ☐ WAF rules active (rate limiting, injection protection)
  ☐ No public IP on application pods
  ☐ TLS 1.3 enforced end-to-end
  ☐ Penetration test completed (Critical=0, High=0)

Monitoring:
  ☐ CloudWatch/Splunk integration active
  ☐ PII scanning on audit logs (regex-based alert)
  ☐ LLM API cost alerts configured
  ☐ Incident response runbook rehearsed

Responsible AI:
  ☐ Model card completed (intended use, limitations, failure modes)
  ☐ Hallucination rate measured and documented (<5% threshold)
  ☐ Human escalation path configured ("contact your Account Executive")
  ☐ Quarterly accuracy review cadence scheduled
```

### ML-Specific Incident Response

| Incident | Trigger | Response |
|---|---|---|
| Hallucination detected | User reports incorrect answer | 1. Log to incident tracker 2. Identify source document gap 3. Add corrective document or retune prompt |
| PII surfaced in response | Any personal data in bot answer | 1. Immediately disable bot 2. Identify source PDF 3. Remove PII document 4. Re-index 5. Notify CISO |
| LLM API key compromise | Unauthorised API usage detected | 1. Rotate key immediately via Secrets Manager 2. Review CloudTrail logs 3. Notify CISO |
| Vector store corruption | Irrelevant/nonsensical answers | 1. Delete and rebuild vector store from clean PDFs 2. Run validation test suite |
| Adversarial prompt injection | Jailbreak attempt detected | 1. Log attempt 2. Review system prompt robustness 3. Add input sanitisation 4. Alert security team |

---

## Quick Reference

```bash
# Run locally
python -m src.app

# Run tests
pytest tests/ -v

# Build Docker image
docker build -t account-manager-bot:latest .

# Run in Docker
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=sk-... \
  -v $(pwd)/docs:/app/docs \
  account-manager-bot:latest

# Kubernetes deploy (production)
helm upgrade --install account-manager-bot ./helm \
  --namespace account-bot \
  --set image.tag=latest \
  --values helm/values.prod.yaml

# Rollback
helm rollback account-manager-bot -n account-bot
```

---

*Account Manager Bot — LangChain + LLM document QA. Not a substitute for official company policy.*
