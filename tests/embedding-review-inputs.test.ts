import { describe, expect, it, vi } from "vitest";

vi.mock("../src/onboarding/validate-provider.js", () => ({
  validateProvider: vi.fn(async () => ({ ok: true, dimensions: 1024, encodingFormat: "float" })),
  validateLocal: vi.fn(),
}));

describe("embedding configuration input validation", () => {
  it("rejects an explicit base64 setting that cannot force SDK decoding", async () => {
    const { validateScrybeConfig } = await import("../src/config.js");
    expect(validateScrybeConfig({
      schema_version: 1,
      embedding_presets: {
        custom: { provider: "custom", model: "qwen", base_url: "http://localhost/v1", dim: 1024, encoding_format: "base64" },
      },
      assignments: { code_preset: "custom", text_preset: "custom" },
    })).toContain('encoding_format must be "float"');
  });

  it.each([
    { code_provider: "local", code_encoding_format: "float" as const },
    { code_provider: "local", text_provider: "local", text_encoding_format: "float" as const },
  ])("rejects encoding for a non-custom init provider %j", async (input) => {
    const { initTool } = await import("../src/tools/init-mcp.js");
    const result = await initTool.handler(input);
    expect(result.ok).toBe(false);
    expect(result.error).toContain("only valid");
  });

  it("infers dimensions and persists float encoding during custom init", async () => {
    const { initTool } = await import("../src/tools/init-mcp.js");
    const result = await initTool.handler({
      code_provider: "custom", code_model: "qwen", code_base_url: "http://localhost/v1", code_api_key: "not-needed",
      text_provider: "local",
    });
    expect(result).toMatchObject({ ok: true, status: "configured" });
    const { readScrybeConfig } = await import("../src/config.js");
    const cfg = readScrybeConfig()!;
    expect(cfg.embedding_presets[cfg.assignments.code_preset]).toMatchObject({ dim: 1024, encoding_format: "float" });
  });

  it("rejects a supplied dimension that disagrees with the init probe", async () => {
    const { initTool } = await import("../src/tools/init-mcp.js");
    const result = await initTool.handler({
      code_provider: "custom", code_model: "qwen", code_base_url: "http://localhost/v1", code_dim: 512, code_api_key: "not-needed",
      text_provider: "local",
    });
    expect(result).toMatchObject({ ok: false, status: "validation_failed" });
    const { readScrybeConfig } = await import("../src/config.js");
    expect(readScrybeConfig()).toBeNull();
  });
});
