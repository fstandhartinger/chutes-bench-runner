# AIME 2025 benchmark identity

Last verified: 2026-08-07.

The `aime_2025` adapter loads the 15 problems from AIME 2025 I followed by
the 15 problems from AIME 2025 II. Its source is
[`opencompass/AIME2025` revision `a6ad95f6`](https://huggingface.co/datasets/opencompass/AIME2025/tree/a6ad95f611d72cf628a80b58bd0432ef6638f958),
using the `AIME2025-I` and `AIME2025-II` configurations and their `test`
splits. Runtime preload requires 15 rows from each configuration and compares
the ordered synthetic task IDs `2025-I-01..15, 2025-II-01..15` to the pinned
30-item manifest.

The previous loader used the mutable `train` split of
`AI-MO/aimo-validation-aime`. A real preload on 2026-08-07 produced 90 rows:
30 each from 2022, 2023, and 2024, and no 2025 problem. That source therefore
did not implement the adapter's name and is no longer used.

`tests/test_aime_adapter.py::test_pinned_aime_2025_registry_loader_is_exact`
constructs the adapter through the registry, runs `preload()`, and checks the
count, uniqueness, ordered task IDs, and dataset revision.
