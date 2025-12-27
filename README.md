# Chutes Bench Runner

A production-grade LLM benchmark runner for models hosted on [Chutes](https://chutes.ai). Run comprehensive benchmark suites with one click from a modern web UI.

![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Features

- **One-Click Benchmarking**: Select a model, choose subset size, and run 11 industry-standard benchmarks
- **Deterministic Subsampling**: Run 1%, 5%, 10%, 25%, 50%, or 100% of benchmarks with reproducible results
- **Live Progress**: Real-time streaming updates via SSE (Server-Sent Events)
- **Detailed Results**: Per-benchmark and per-item results with metrics, latencies, and scoring
- **Exports**: Download results as CSV or PDF reports
- **Modern UI**: Dark mode, minimalistic, futuristic Apple-style design

## Supported Benchmarks

| Benchmark | Description | Setup Required |
|-----------|-------------|----------------|
| MMLU-Pro | Multi-task language understanding (57 subjects) | No |
| GPQA Diamond | Graduate-level science questions | No |
| Humanity's Last Exam | Expert-level questions across domains | Yes |
| LiveCodeBench | Live programming problems | Yes |
| SciCode | Scientific computing code generation | Yes |
| AIME 2025 | Math competition problems | No |
| IFBench | Instruction following evaluation | No |
| AA-LCR | Adversarial code reasoning | Yes |
| Terminal-Bench Hard | CLI/terminal interaction | Yes |
| τ²-Bench Telecom | Telecommunications agent benchmark | Yes |
| SWE-Bench Pro | Real GitHub issue resolution | Yes |

## Architecture

```
chutes-bench-runner/
├── backend/           # Python + FastAPI backend
│   ├── app/
│   │   ├── api/       # REST API endpoints
│   │   ├── benchmarks/# Benchmark adapters
│   │   ├── core/      # Config, logging
│   │   ├── db/        # Database session
│   │   ├── models/    # SQLAlchemy models
│   │   ├── services/  # Business logic
│   │   └── worker/    # Background worker
│   ├── alembic/       # Database migrations
│   └── tests/         # Backend tests
├── frontend/          # Next.js + TypeScript frontend
│   ├── app/           # App router pages
│   ├── components/    # React components
│   ├── lib/           # API client, utilities
│   └── tests/         # Frontend tests
├── scripts/           # Development scripts
├── docker-compose.yml # Local development
└── render.yaml        # Render deployment
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- PostgreSQL (or Neon account)
- Chutes API key

### Local Development

1. **Clone and setup**:
   ```bash
   git clone https://github.com/fstandhartinger/chutes-bench-runner.git
   cd chutes-bench-runner
   ./scripts/bootstrap.sh
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your credentials:
   # - DATABASE_URL: Your Neon/Postgres connection string
   # - CHUTES_API_KEY: Your Chutes API key
   ```

3. **Run migrations**:
   ```bash
   cd backend
   source .venv/bin/activate
   alembic upgrade head
   ```

4. **Sync models from Chutes**:
   ```bash
   python ../scripts/seed_models.py
   ```

5. **Start development servers**:
   ```bash
   ./scripts/dev.sh
   ```

6. **Open the app**:
   - Frontend: http://localhost:3000
   - API Docs: http://localhost:8000/docs

### Using Docker Compose

```bash
# Start all services
CHUTES_API_KEY=your_key docker-compose up

# Access the app at http://localhost:3000
```

## Deterministic Subset Sampling

Subset sampling is a first-class feature that enables:
- **Fast iteration**: Test with 1-10% of data during development
- **Reproducibility**: Same seed produces identical item selection
- **Scalability**: Full runs at 100% for final evaluation

### How It Works

1. Each run has a unique ID used as part of the random seed
2. The seed is combined with benchmark name: `sha256(run_id + benchmark_name)`
3. Items are shuffled deterministically and top K are selected
4. Sampled item IDs are persisted for audit and reproducibility

```python
# Example: 10% subset of 1000 items
seed = sha256(f"{run_id}_{benchmark_name}").hexdigest()[:16]
rng = random.Random(int(seed, 16))
shuffled = items.copy()
rng.shuffle(shuffled)
sampled = shuffled[:100]  # 10% of 1000
```

## Adding a New Benchmark Adapter

1. Create adapter file in `backend/app/benchmarks/adapters/`:

```python
from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter

@register_adapter("my_benchmark")
class MyBenchmarkAdapter(BenchmarkAdapter):
    def get_name(self) -> str:
        return "my_benchmark"
    
    def get_display_name(self) -> str:
        return "My Benchmark"
    
    async def get_total_items(self) -> int:
        # Return total number of items
        pass
    
    async def enumerate_items(self):
        # Yield item IDs
        pass
    
    async def evaluate_item(self, item_id: str) -> ItemResult:
        # Evaluate single item and return result
        pass
```

2. Add to `__init__.py` imports
3. Create migration to seed benchmark metadata
4. Implement `preload()` for dataset loading

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/models` | List available Chutes models |
| GET | `/api/benchmarks` | List available benchmarks |
| POST | `/api/runs` | Create new benchmark run |
| GET | `/api/runs` | List runs with filters |
| GET | `/api/runs/{id}` | Get run details |
| GET | `/api/runs/{id}/events` | SSE stream of run events |
| POST | `/api/runs/{id}/cancel` | Cancel running/queued run |
| GET | `/api/runs/{id}/export` | Export as CSV or PDF |
| POST | `/api/admin/sync-models` | Refresh models from Chutes |

## Deployment to Render

1. Push to GitHub:
   ```bash
   git push origin main
   ```

2. Deploy via Render Dashboard or CLI:
   ```bash
   render deploy
   ```

3. Configure environment variables in Render:
   - `CHUTES_API_KEY`: Your Chutes API key
   - `ADMIN_SECRET`: Secret for admin endpoints

The `render.yaml` Blueprint will create:
- Web service for backend API
- Background worker for benchmark execution
- Web service for frontend
- PostgreSQL database

## Security Notes

- **API Keys**: Never commit API keys. Use environment variables.
- **Admin Secret**: Protect admin endpoints with `X-Admin-Secret` header
- **Database**: Use SSL connections (Neon provides this by default)
- **CORS**: Configure allowed origins for production

## Testing

### Backend Tests
```bash
cd backend
source .venv/bin/activate
pytest -v
```

### Frontend Tests
```bash
cd frontend
npm test
```

### E2E Tests
```bash
cd frontend
npx playwright test
```

## Demo Flow

1. **Select a model** from the dropdown (models are synced from Chutes)
2. **Choose subset size** (start with 5-10% for quick testing)
3. **Toggle benchmarks** to include/exclude specific ones
4. **Click "Run Benchmarks"** to start
5. **Watch live progress** as benchmarks complete
6. **View results** in the run detail page
7. **Export** as CSV or PDF for analysis

## Troubleshooting

### Models not appearing
Run the model sync:
```bash
curl -X POST http://localhost:8000/api/admin/sync-models \
  -H "X-Admin-Secret: your_secret"
```

### Worker not processing runs
Check worker logs for database connectivity issues:
```bash
docker-compose logs worker
```

### Benchmark shows "needs_setup"
Some benchmarks require additional dependencies or gated dataset access. Check the `setup_notes` field for requirements.

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

---

Built with ❤️ for the [Chutes](https://chutes.ai) ecosystem.

