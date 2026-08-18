/**
 * Regression tests for semantic-search fan-out.
 *
 * The vector store is mocked because it is an external database boundary; the
 * real search pipeline, source filtering, and cross-source result merging run
 * unchanged. Removing the query-embedding cache makes the first two tests
 * issue one embedding request per source instead of one per resolved config.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const state = vi.hoisted(() => ({
  project: undefined as any,
  configs: new Map<string, any>(),
  results: new Map<string, Array<{ chunk_id: string; content: string; score: number }>>(),
}));
const initialSkipMigration = process.env.SCRYBE_SKIP_MIGRATION;

vi.mock("../src/config.js", () => ({
  config: {
    rerankEnabled: false,
    rerankFetchMultiplier: 3,
    hybridEnabled: false,
    rrfK: 60,
  },
}));

vi.mock("../src/registry.js", () => ({
  getProject: vi.fn(() => state.project),
  resolveEmbeddingConfig: vi.fn((source: { source_id: string }) => state.configs.get(source.source_id)),
}));

vi.mock("../src/plugins/index.js", () => ({
  getPlugin: vi.fn((type: string) => ({ embeddingProfile: type === "ticket" ? "text" : "code" })),
}));

vi.mock("../src/embedder.js", () => ({
  embedQuery: vi.fn(async (_query: string, config: { dimensions: number }) => [config.dimensions]),
}));

vi.mock("../src/vector-store.js", () => ({
  search: vi.fn(async (_query: number[], _projectId: string, _limit: number, tableName: string) => {
    const configuredResults = state.results.get(tableName);
    if (configuredResults) return configuredResults;
    const score = tableName === "table_ionic" ? 0.91 : 0.42;
    return [{ chunk_id: `code-${tableName}`, content: tableName, score }];
  }),
  ftsSearch: vi.fn(),
  searchKnowledge: vi.fn(async (_query: number[], _projectId: string, _limit: number, tableName: string) => [
    { project_id: "project", source_id: "", item_path: tableName, content: tableName, item_type: "document" },
  ]),
  ftsSearchKnowledge: vi.fn(),
}));

vi.mock("../src/reranker.js", () => ({ rerank: vi.fn() }));

vi.mock("../src/branch-state.js", () => ({
  resolveBranch: vi.fn(),
  getChunkIdsForBranch: vi.fn(),
  getBranchesForChunks: vi.fn(() => new Map()),
  resolveBranchForSearch: vi.fn(),
}));

vi.mock("../src/daemon/caller-error.js", () => ({ markCallerFacing: <T extends Error>(error: T) => error }));

function source(sourceId: string, type: "code" | "ticket") {
  return {
    source_id: sourceId,
    source_config: { type, root_path: `/fixtures/${sourceId}`, languages: [] },
    table_name: `table_${sourceId}`,
  };
}

function embeddingConfig(overrides: Record<string, unknown> = {}) {
  return {
    base_url: "http://127.0.0.1:11480/v1",
    model: "qwen3-embedding",
    dimensions: 1024,
    api_key_env: "SCRYBE_VLLM_API_KEY",
    provider_type: "api" as const,
    prompt_template: { query: "query: ", passage: "passage: " },
    max_input_tokens: 32768,
    ...overrides,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  state.project = undefined;
  state.configs.clear();
  state.results.clear();
  process.env.SCRYBE_SKIP_MIGRATION = "1";
});

afterEach(() => {
  if (initialSkipMigration === undefined) {
    delete process.env.SCRYBE_SKIP_MIGRATION;
  } else {
    process.env.SCRYBE_SKIP_MIGRATION = initialSkipMigration;
  }
});

describe("searchCode query embedding fan-out", () => {
  it("shares one embedding across code sources with an equal resolved configuration", async () => {
    state.project = { id: "code-project", sources: [source("one", "code"), source("two", "code"), source("three", "code")] };
    for (const sourceId of ["one", "two", "three"]) state.configs.set(sourceId, embeddingConfig());

    const { searchCode } = await import("../src/search.js");
    const { embedQuery } = await import("../src/embedder.js");
    const results = await searchCode("find API authentication", "code-project");

    expect(results.map((result) => result.source_id).sort()).toEqual(["one", "three", "two"]);
    expect(embedQuery).toHaveBeenCalledTimes(1);
  });

  it("keeps embeddings separate when a resolved configuration changes", async () => {
    state.project = { id: "code-project", sources: [source("one", "code"), source("two", "code"), source("three", "code")] };
    state.configs.set("one", embeddingConfig());
    state.configs.set("two", embeddingConfig({ api_key_env: "SECOND_LOCAL_PROVIDER_KEY" }));
    state.configs.set("three", embeddingConfig({ prompt_template: { query: "code query: ", passage: "code passage: " } }));

    const { searchCode } = await import("../src/search.js");
    const { embedQuery } = await import("../src/embedder.js");
    const results = await searchCode("find API authentication", "code-project");

    expect(results).toHaveLength(3);
    expect(embedQuery).toHaveBeenCalledTimes(3);
  });

  it("keeps embeddings separate when only the response encoding changes", async () => {
    state.project = { id: "code-project", sources: [source("one", "code"), source("two", "code")] };
    state.configs.set("one", embeddingConfig({ encoding_format: "float" }));
    state.configs.set("two", embeddingConfig({ encoding_format: "base64" }));

    const { searchCode } = await import("../src/search.js");
    const { embedQuery } = await import("../src/embedder.js");
    const results = await searchCode("find API authentication", "code-project");

    expect(results).toHaveLength(2);
    expect(embedQuery).toHaveBeenCalledTimes(2);
  });

  it("does not retain a query vector after the search completes", async () => {
    state.project = { id: "code-project", sources: [source("one", "code"), source("two", "code")] };
    for (const sourceId of ["one", "two"]) state.configs.set(sourceId, embeddingConfig());

    const { searchCode } = await import("../src/search.js");
    const { embedQuery } = await import("../src/embedder.js");

    await searchCode("find API authentication", "code-project");
    await searchCode("find API authentication", "code-project");

    expect(embedQuery).toHaveBeenCalledTimes(2);
  });

  it("ranks equal-config sources by semantic score instead of source order", async () => {
    state.project = {
      id: "code-project",
      sources: [source("cmx", "code"), source("node-services", "code"), source("ionic", "code")],
    };
    for (const sourceId of ["cmx", "node-services", "ionic"]) state.configs.set(sourceId, embeddingConfig());

    const { searchCode } = await import("../src/search.js");
    const results = await searchCode("find feedback widget PII", "code-project");

    expect(results.map((result) => result.source_id)).toEqual(["ionic", "cmx", "node-services"]);
  });

  it("interleaves equal-config source results by semantic score", async () => {
    state.project = {
      id: "code-project",
      sources: [source("cmx", "code"), source("ionic", "code")],
    };
    for (const sourceId of ["cmx", "ionic"]) state.configs.set(sourceId, embeddingConfig());
    state.results.set("table_cmx", [
      { chunk_id: "cmx-high", content: "cmx high", score: 0.8 },
      { chunk_id: "cmx-low", content: "cmx low", score: 0.4 },
    ]);
    state.results.set("table_ionic", [
      { chunk_id: "ionic-high", content: "ionic high", score: 0.91 },
      { chunk_id: "ionic-low", content: "ionic low", score: 0.55 },
    ]);

    const { searchCode } = await import("../src/search.js");
    const results = await searchCode("find feedback widget PII", "code-project");

    expect(results.map((result) => result.chunk_id)).toEqual(["ionic-high", "cmx-high", "ionic-low", "cmx-low"]);
  });

  it("keeps rank fusion when source embedding configurations differ", async () => {
    state.project = {
      id: "code-project",
      sources: [source("cmx", "code"), source("ionic", "code")],
    };
    state.configs.set("cmx", embeddingConfig());
    state.configs.set("ionic", embeddingConfig({ model: "other-embedding-model" }));

    const { searchCode } = await import("../src/search.js");
    const results = await searchCode("find feedback widget PII", "code-project");

    expect(results.map((result) => result.source_id)).toEqual(["cmx", "ionic"]);
  });

  it("ignores an unresolved branch from a differently configured source when ranking matches", async () => {
    state.project = {
      id: "code-project",
      sources: [source("cmx", "code"), source("ionic", "code"), source("unmatched", "code")],
    };
    state.configs.set("cmx", embeddingConfig());
    state.configs.set("ionic", embeddingConfig());
    state.configs.set("unmatched", embeddingConfig({ model: "other-embedding-model" }));
    process.env.SCRYBE_SKIP_MIGRATION = "0";

    const branchState = await import("../src/branch-state.js");
    vi.mocked(branchState.resolveBranchForSearch).mockImplementation((_projectId, sourceId) =>
      sourceId === "unmatched" ? null : "main"
    );
    vi.mocked(branchState.getChunkIdsForBranch).mockReturnValue(new Set());

    const { searchCode } = await import("../src/search.js");
    const results = await searchCode("find feedback widget PII", "code-project", { branch: "main" });

    expect(results.map((result) => result.source_id)).toEqual(["ionic", "cmx"]);
  });

  it("returns no results when every code source is unindexed", async () => {
    const unindexed = source("unindexed", "code");
    delete unindexed.table_name;
    state.project = { id: "code-project", sources: [unindexed] };

    const { searchCode } = await import("../src/search.js");

    await expect(searchCode("find feedback widget PII", "code-project")).resolves.toEqual([]);
  });

  it("globally ranks hybrid vector results before fusing full-text results", async () => {
    state.project = {
      id: "code-project",
      sources: [source("cmx", "code"), source("ionic", "code")],
    };
    for (const sourceId of ["cmx", "ionic"]) state.configs.set(sourceId, embeddingConfig());

    const configModule = await import("../src/config.js");
    const vectorStore = await import("../src/vector-store.js");
    configModule.config.hybridEnabled = true;
    vi.mocked(vectorStore.ftsSearch).mockImplementation(async (_query, _projectId, _limit, tableName) => [
      { chunk_id: `fts-${tableName}`, content: "full text match", score: 0 },
    ] as any);

    try {
      const { searchCode } = await import("../src/search.js");
      const results = await searchCode("find feedback widget PII", "code-project");

      expect(results.map((result) => result.chunk_id)).toEqual([
        "code-table_ionic",
        "fts-table_cmx",
        "code-table_cmx",
        "fts-table_ionic",
      ]);
    } finally {
      configModule.config.hybridEnabled = false;
      vi.mocked(vectorStore.ftsSearch).mockReset();
    }
  });
});

describe("searchKnowledge query embedding fan-out", () => {
  it("shares one embedding across knowledge sources with an equal resolved configuration", async () => {
    state.project = { id: "knowledge-project", sources: [source("one", "ticket"), source("two", "ticket"), source("three", "ticket")] };
    for (const sourceId of ["one", "two", "three"]) state.configs.set(sourceId, embeddingConfig());

    const { searchKnowledge } = await import("../src/search.js");
    const { embedQuery } = await import("../src/embedder.js");
    const results = await searchKnowledge("find rollout notes", "knowledge-project", 10);

    expect(results).toHaveLength(3);
    expect(embedQuery).toHaveBeenCalledTimes(1);
  });
});
