"""Immutable identity for the public DeepSWE v1.1 task corpus."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeepSWESpec:
    """A pinned, reviewable DeepSWE release."""

    display_name: str
    expected_count: int
    repository: str
    commit: str
    archive_sha256: str
    task_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.task_ids) != self.expected_count:
            raise ValueError(
                f"{self.display_name} manifest has {len(self.task_ids)} task IDs; "
                f"the named release requires {self.expected_count}"
            )
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError(f"{self.display_name} manifest contains duplicate task IDs")

    @property
    def archive_url(self) -> str:
        return f"https://codeload.github.com/{self.repository}/tar.gz/{self.commit}"


# Source: the 113 task directories under tasks/ at the pinned official
# datacurve-ai/deep-swe commit below. That revision identifies the corpus as
# v1.1 in every task image tag and uses schema 1.3's separate verifier mode.
DEEPSWE_V1_1_TASK_IDS = (
    "abs-module-cache-flags",
    "abs-stepped-slices",
    "actionlint-action-pinning-lint",
    "adaptix-name-mapping-aliases",
    "aiomonitor-task-snapshots-diff",
    "anko-default-function-arguments",
    "anko-typed-variable-bindings",
    "arcane-drift-detection-baselines",
    "arktype-json-schema-refs-dependencies",
    "awilix-async-container-initialization",
    "bandit-incremental-cache-control",
    "bandit-interprocedural-taint-checks",
    "bandit-structured-nosec-directives",
    "boa-hierarchical-evaluation-cancellation",
    "cattrs-partial-structuring-recovery",
    "clack-async-autocomplete-options",
    "claude-code-by-agents-recursive-delegation",
    "cliffy-config-file-parsing",
    "csstree-shorthand-expansion-compression",
    "dasel-html-document-format",
    "dateutil-rfc5545-timezone-interop",
    "drizzle-orm-window-function-builders",
    "dynamodb-toolbox-conditional-attribute-requirements",
    "dynamodb-toolbox-lazy-recursive-schemas",
    "effect-sse-httpapi-streaming",
    "eicrud-keyset-pagination-cursor",
    "etree-xml-diff-patch",
    "expr-try-catch-errors",
    "fastapi-deprecation-response-headers",
    "fastapi-implicit-head-options",
    "fd-deterministic-multi-key-sorting",
    "geo-shapeindex-serialization",
    "go-critic-doc-link-checker",
    "go-genai-streamed-function-args",
    "go-git-worktree-merge-conflicts",
    "goreleaser-retry-publish-auditing",
    "gql-incremental-graphql-delivery",
    "happy-dom-abort-pending-body-reads",
    "happy-dom-deterministic-intersectionobserver",
    "helm-array-merge-strategies",
    "helm-unified-manifest-stream",
    "httpx-deterministic-cookie-store",
    "httpx-multipart-response-parsing",
    "httpx-streaming-json-iteration",
    "igel-persist-feature-schema",
    "ink-grid-box-layout",
    "ipython-session-bundle-replay",
    "katex-multicolumn-array-spans",
    "kcp-go-multiplexed-kcp-streams",
    "kea-atomic-signal-selectors",
    "kgateway-consistent-hash-policy",
    "kombu-single-active-consumer-priority",
    "kombu-virtual-queue-dead-lettering",
    "koota-composite-trait-aspects",
    "koota-deferred-mutation-buffer",
    "koota-entity-snapshot-rollback",
    "koota-pair-relation-tracking",
    "koota-query-predicates",
    "kysely-window-grouping-helpers",
    "langchain-request-coalescing",
    "mashumaro-flattened-dataclass-fields",
    "meriyah-explicit-resource-declarations",
    "mnamer-daemon-watch-lifecycle",
    "mobly-grouped-test-barriers",
    "narwhals-rolling-window-suite",
    "numba-stencil-boundary-modes",
    "obsidian-linter-auto-table-of-contents",
    "obsidian-linter-link-format-conversion",
    "obsidian-linter-scoped-ignore-markers",
    "ofetch-per-origin-circuit-breaker",
    "onedump-dump-encryption-pipeline",
    "opa-rego-rule-profiling",
    "opa-template-string-reconstruction",
    "optique-conditional-option-dependencies",
    "oxvg-structural-selector-preservation",
    "participle-grammar-conflict-analysis",
    "pebble-durability-wait-apis",
    "pest-character-class-coalescing",
    "prometheus-transactional-reload-status",
    "prometheus-typed-label-sorting",
    "psd-tools-blend-range-api",
    "pwntools-tube-multiplexing",
    "python-statemachine-state-data-scoping",
    "query-persist-restored-query-state",
    "quill-shared-toolbar-focus",
    "returns-validated-error-accumulation",
    "scc-bounded-memory-spilling",
    "scriggo-method-declarations",
    "skrub-duration-encoding",
    "sql-formatter-bigquery-pipe-formatting",
    "sqlfmt-create-table-ddl-formatting",
    "sqlite-utils-safe-import-checkpoints",
    "superjson-error-stack-serialization",
    "task-task-graph-export",
    "tengo-callable-instance-isolation",
    "tengo-destructuring-bindings",
    "termenv-preserve-ansi-resets",
    "testem-bail-on-test-failure",
    "testem-per-launcher-reports",
    "textual-kitty-key-phases",
    "textual-richlog-follow-state",
    "tomlkit-toml-table-converters",
    "true-myth-iterable-collection-combinators",
    "ts-pattern-match-each",
    "updo-policy-alerting",
    "valibot-recursive-schema-composition",
    "vitest-duration-sharding",
    "vulture-persistent-analysis-cache",
    "wasmi-trap-coredumps",
    "wazero-multi-module-snapshots",
    "yaegi-go-embed-directives",
    "yjs-map-conflict-detection",
    "ytt-jsonpath-query-api",
)


DEEPSWE_V1_1 = DeepSWESpec(
    display_name="DeepSWE v1.1 (Sandy CLI scaffold)",
    expected_count=113,
    repository="datacurve-ai/deep-swe",
    commit="435ee89ec2f2e2289f33b0da4f992f0b7b7266b9",
    archive_sha256="34c6fabd3dad1770d753829378a81c3d8bb658ff255de9f01f3606e213cd2b46",
    task_ids=DEEPSWE_V1_1_TASK_IDS,
)
