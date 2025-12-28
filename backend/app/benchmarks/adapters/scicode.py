"""SciCode benchmark adapter."""
import time
import re
from typing import Any, AsyncIterator, Optional

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.core.logging import get_logger
from app.services.sandy_service import SandyService

logger = get_logger(__name__)


@register_adapter("scicode")
class SciCodeAdapter(BenchmarkAdapter):
    """
    SciCode benchmark adapter.
    
    Scientific computing code generation benchmark.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._items: list[dict[str, Any]] = []
        self.sandy = SandyService()

    def get_name(self) -> str:
        return "scicode"

    def get_display_name(self) -> str:
        return "SciCode"

    def requires_setup(self) -> bool:
        return True

    def get_setup_notes(self) -> Optional[str]:
        return "SciCode requires scientific computing libraries (numpy, scipy, etc.) for execution."

    async def get_total_items(self) -> int:
        if not self._items:
            await self.preload()
        return len(self._items)

    async def preload(self) -> None:
        """Load SciCode dataset."""
        if self._items:
            return

        try:
            from datasets import load_dataset
            import os

            logger.info("Loading SciCode dataset")
            hf_token = os.environ.get("HF_TOKEN")
            
            dataset = None
            # Try to load from SciCode or alternatives
            sources = [
                ("SciCode-Bench/SciCode", "test", {}),
                ("bigcode/humanevalpack", "test", {"name": "python"}),
            ]
            
            for source_name, split, extra_kwargs in sources:
                try:
                    logger.info(f"Trying to load from {source_name}")
                    kwargs = {"token": hf_token} if hf_token else {}
                    kwargs.update(extra_kwargs)
                    dataset = load_dataset(source_name, split=split, **kwargs)
                    break
                except Exception as e:
                    logger.warning(f"Could not load {source_name}: {e}")
                    continue
            
            if dataset is None:
                # Use placeholder scientific computing problems
                self._items = [
                    {"id": "0", "problem": "Write a NumPy function to compute the eigenvalues of a symmetric matrix.", "domain": "linear algebra", "context": "import numpy as np", "test": "import numpy as np\nA = np.array([[2, 1], [1, 2]])\nevals = compute_eigenvalues(A)\nassert np.allclose(sorted(evals), [1, 3])"},
                    {"id": "1", "problem": "Implement numerical integration using Simpson's rule.", "domain": "numerical methods", "context": "import numpy as np", "test": "import numpy as np\nf = lambda x: x**2\nresult = simpsons_rule(f, 0, 1, 100)\nassert np.isclose(result, 1/3, atol=1e-3)"},
                    {"id": "2", "problem": "Write a function to solve a system of ODEs using Runge-Kutta 4th order method.", "domain": "differential equations", "context": "import numpy as np", "test": "import numpy as np\ndef f(t, y): return -y\ny = solve_ode(f, 0, 1, 0.1, 10)\nassert np.isclose(y[-1], np.exp(-1), atol=1e-2)"},
                ]
                logger.info(f"Using {len(self._items)} placeholder SciCode items")
                return
            
            self._items = []
            for i, item in enumerate(dataset):
                problem = item.get("problem_description") or item.get("prompt") or item.get("question") or ""
                test = item.get("test", "") or item.get("canonical_solution", "")
                if problem:
                    self._items.append({
                        "id": str(i),
                        "problem": problem,
                        "domain": item.get("domain", ""),
                        "context": item.get("context", ""),
                        "test": test,
                    })
            
            logger.info(f"Loaded {len(self._items)} SciCode items")
        except Exception as e:
            logger.error("Failed to load SciCode", error=str(e))
            self._items = []

    async def enumerate_items(self) -> AsyncIterator[str]:
        if not self._items:
            await self.preload()
        for item in self._items:
            yield item["id"]

    async def evaluate_item(self, item_id: str) -> ItemResult:
        """Evaluate a single SciCode item."""
        if not self._items:
            await self.preload()

        item = next((i for i in self._items if i["id"] == item_id), None)
        if not item:
            return ItemResult(item_id=item_id, error=f"Item {item_id} not found")

        context = item.get("context", "")
        prompt = f"""Solve the following scientific computing problem. Write Python code using numpy/scipy.

{f"Context: {context}" if context else ""}

Problem: {item["problem"]}

Solution:
```python
"""

        try:
            start_time = time.time()
            response_text, metadata = await self.client.get_completion_text(
                self.model_slug,
                prompt,
                system_prompt="You are an expert scientific programmer. Write efficient, correct numerical code. Be concise and provide ONLY the code block.",
                max_tokens=4096,
                temperature=0.0,
            )
            latency_ms = int((time.time() - start_time) * 1000)

            # Extract code from response robustly
            extracted_code = self.extract_python_code(response_text)
            
            # Prepare execution code
            test_code = item.get("test", "")
            full_code = f"{context}\n\n{extracted_code}\n\n{test_code}"
            
            # Execute in sandbox
            execution_result = await self.sandy.run_python_code(full_code)
            
            is_correct = execution_result.get("success", False) and execution_result.get("exit_code") == 0
            error = None
            if not is_correct:
                error = execution_result.get("stderr") or execution_result.get("error")

            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(item["problem"]),
                prompt=prompt,
                response=response_text.strip(),
                expected="[Tests passed]",
                is_correct=is_correct,
                score=1.0 if is_correct else 0.0,
                latency_ms=latency_ms,
                input_tokens=metadata.get("usage", {}).get("prompt_tokens"),
                output_tokens=metadata.get("usage", {}).get("completion_tokens"),
                metadata={"domain": item.get("domain")},
                judge_output={
                    "stdout": execution_result.get("stdout"),
                    "stderr": execution_result.get("stderr"),
                    "exit_code": execution_result.get("exit_code")
                },
                error=error
            )

        except Exception as e:
            logger.error("SciCode evaluation failed", item_id=item_id, error=str(e))
            return ItemResult(item_id=item_id, prompt=prompt, error=str(e))


