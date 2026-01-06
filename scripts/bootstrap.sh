#!/bin/bash
# Bootstrap script for local development

set -e

echo "🚀 Bootstrapping Chutes Bench Runner..."

# Check for required tools
command -v python3 >/dev/null 2>&1 || { echo "Python 3 is required"; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "npm is required"; exit 1; }

# Backend setup
echo "📦 Setting up backend..."
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,benchmarks]"
cd ..

# Frontend setup
echo "📦 Setting up frontend..."
cd frontend
npm install
cd ..

echo ""
echo "✅ Bootstrap complete!"
echo ""
echo "Next steps:"
echo "1. Copy .env.example to .env and configure:"
echo "   - DATABASE_URL (Neon connection string)"
echo "   - CHUTES_API_KEY (your Chutes API key)"
echo ""
echo "2. Run migrations:"
echo "   cd backend && source .venv/bin/activate && alembic upgrade head"
echo ""
echo "3. Start development servers:"
echo "   ./scripts/dev.sh"
echo ""


















