# Schedule Risk Analyser — nPlan-Style Python Service

## What This Is

A realistic Python microservice that mirrors what you'd build at nPlan as an AI Engineer.
It's a **FastAPI** service that takes construction schedule data and uses an LLM to
analyse risks — similar to how nPlan's **Barry** AI assistant works.

This gives you a working reference for how Python, FastAPI, Pydantic, pytest, Docker,
and Azure all fit together in a real AI engineering workflow.

---

## How This Maps to nPlan's Stack

| This Example               | nPlan Equivalent                              |
|-----------------------------|-----------------------------------------------|
| FastAPI service             | Their Python API backends                     |
| Pydantic models             | Data validation for schedule inputs           |
| LLM integration (Anthropic) | Barry AI assistant / Schedule Studio AI       |
| pytest tests                | Their eval/testing framework                  |
| Docker container            | Azure Container Apps deployment               |
| Structured prompts          | Domain-grounded prompts for construction AI   |

---

## Project Structure

```
nplan-example/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app — entry point
│   ├── models.py            # Pydantic data models (schedule, risks)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── risk_engine.py   # Core risk analysis logic
│   │   └── llm_client.py    # LLM integration (Anthropic API)
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── templates.py     # Structured prompts (like your trade profiles)
│   └── config.py            # Settings / environment config
├── tests/
│   ├── __init__.py
│   ├── test_models.py       # Unit tests for data models
│   ├── test_risk_engine.py  # Unit tests for risk logic
│   └── test_api.py          # Integration tests for API endpoints
├── Dockerfile                # Container for Azure deployment
├── docker-compose.yml        # Local dev with all services
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Modern Python project config
└── README.md                 # This file
```

---

## Quick Start (Run Locally)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key (for LLM features)
export ANTHROPIC_API_KEY="sk-ant-..."

# 4. Run the dev server
uvicorn app.main:app --reload --port 8000

# 5. Open the auto-generated API docs
# http://localhost:8000/docs     (Swagger UI — interactive testing)
# http://localhost:8000/redoc    (ReDoc — cleaner documentation)
```

---

## Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=app --cov-report=term-missing

# Run just the model tests
pytest tests/test_models.py -v
```

---

## Deploy to Azure (How nPlan Does It)

```bash
# 1. Build Docker image
docker build -t schedule-risk-analyser .

# 2. Tag for Azure Container Registry
docker tag schedule-risk-analyser myregistry.azurecr.io/schedule-risk-analyser:latest

# 3. Push to registry
docker push myregistry.azurecr.io/schedule-risk-analyser:latest

# 4. Deploy to Azure Container Apps (their likely deployment target)
az containerapp create \
  --name schedule-risk \
  --resource-group nplan-rg \
  --image myregistry.azurecr.io/schedule-risk-analyser:latest \
  --target-port 8000 \
  --env-vars ANTHROPIC_API_KEY=secretref:anthropic-key
```

---

## Key Concepts for Your Interview

### Why Python (Not PHP/JS)?

Python dominates AI/ML engineering because:
- **Libraries**: numpy, pandas, scikit-learn, pytorch, networkx (graph data!)
- **Type hints + Pydantic**: Strong validation without Java-level boilerplate
- **Async**: FastAPI handles concurrent API requests efficiently
- **ML ecosystem**: All major AI frameworks are Python-first
- **Data science**: Jupyter notebooks for research → FastAPI for production

### Your Transferable Skills

Your PHP REST API endpoints → Python FastAPI routes (same pattern, cleaner syntax)
Your React state management → Pydantic models (structured data validation)
Your AI prompt templates → Same pattern, just in Python instead of JS
Your trade profile guardrails → Same domain grounding, different language

### How You'd Talk About It

"I've been building production APIs in PHP for the ATP platform, and Python's FastAPI
follows the same REST patterns I'm already fluent in — route handlers, request
validation, middleware, async processing. The key difference is Python's ML ecosystem
gives you direct access to tools like networkx for graph manipulation, which maps
perfectly to nPlan's schedule graph data. I've already experimented with Python for
trading bots, and the transition from PHP to Python is far smaller than the transition
from construction worker to AI engineer — which I've already made."
