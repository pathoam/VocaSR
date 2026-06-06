import { describe, expect, it } from "bun:test";
import {
  applyVocaSubstitutions,
  AuthenticatedVocaContextStore,
  BrowserLocalStorageVocaContextStore,
  buildContextPrompt,
  contextOptionsFromUserContext,
  MemoryVocaContextStore,
  normalizeContextTerms,
  parseVocaContextCommand,
  streamConfig,
  validateVocaUserContext,
  VocaClient,
  VocaContextManager,
} from "./index.js";

class TestStorage implements Storage {
  private readonly items = new Map<string, string>();

  get length(): number {
    return this.items.size;
  }

  clear(): void {
    this.items.clear();
  }

  getItem(key: string): string | null {
    return this.items.get(key) ?? null;
  }

  key(index: number): string | null {
    return [...this.items.keys()][index] ?? null;
  }

  removeItem(key: string): void {
    this.items.delete(key);
  }

  setItem(key: string, value: string): void {
    this.items.set(key, value);
  }
}

describe("context helpers", () => {
  it("normalizes context terms and builds the Voca context prompt", () => {
    expect(normalizeContextTerms([" Optagon ", "", "Voca", "Optagon"])).toEqual(["Optagon", "Voca"]);
    expect(buildContextPrompt(["Optagon", "Voca"])).toBe("The following terms may appear in the audio: Optagon, Voca.");
  });

  it("converts a user context into ASR vocabulary terms", () => {
    expect(contextOptionsFromUserContext({
      userId: "user-1",
      context: " Prefer project-specific terms. ",
      vocabulary: [" Optagon ", "Qwen"],
      contacts: [
        { name: " Ada Lovelace ", aliases: [" Countess of Lovelace ", "Ada Lovelace"] },
      ],
      substitutions: [
        { from: "super base", to: "Supabase" },
        { from: "Qwen", to: "Qwen" },
      ],
    })).toEqual({
      context: "Prefer project-specific terms.",
      contextTerms: ["Optagon", "Qwen", "Ada Lovelace", "Countess of Lovelace", "super base", "Supabase"],
    });
  });

  it("can limit upstream context sharing for remote ASR providers", () => {
    const context = {
      userId: "user-1",
      context: "Sensitive project prompt.",
      vocabulary: ["Optagon"],
      contacts: [{ name: "Ada Lovelace", aliases: ["Ada"] }],
      substitutions: [{ from: "super base", to: "Supabase" }],
    };

    expect(contextOptionsFromUserContext(context, "none")).toEqual({});
    expect(contextOptionsFromUserContext(context, "vocabulary_only")).toEqual({
      contextTerms: ["Optagon", "super base", "Supabase"],
    });
  });

  it("validates short request context limits", () => {
    expect(validateVocaUserContext({
      userId: "alice",
      context: "Prefer Optagon frame and Voca terms.",
    }).context).toBe("Prefer Optagon frame and Voca terms.");

    expect(() => validateVocaUserContext({
      userId: "alice",
      context: Array.from({ length: 41 }, (_, index) => `word${index}`).join(" "),
    })).toThrow(/40 words/);
  });

  it("applies substitutions longest-first", () => {
    expect(applyVocaSubstitutions("open super base and get hub", [
      { from: "super", to: "ignored" },
      { from: "super base", to: "Supabase" },
      { from: "get hub", to: "GitHub" },
    ])).toBe("open Supabase and GitHub");
  });

  it("serializes streaming context config", () => {
    expect(streamConfig({ language: "en", token: "t", contextTerms: ["Qwen", "ASR"] })).toEqual({
      language: "en",
      token: "t",
      context_terms: ["Qwen", "ASR"],
    });
  });

  it("parses context-management voice commands", () => {
    expect(parseVocaContextCommand("voca add context", {
      selectedText: "Frame Manager",
    })).toEqual({ type: "add_vocabulary", term: "Frame Manager" });
    expect(parseVocaContextCommand("voice box add contact Ada Lovelace")).toEqual({
      type: "add_contact",
      contact: { name: "Ada Lovelace" },
    });
    expect(parseVocaContextCommand("voca replace super base with Supabase")).toEqual({
      type: "add_substitution",
      from: "super base",
      to: "Supabase",
    });
    expect(parseVocaContextCommand("please add context")).toBeNull();
  });
});

describe("VocaContextManager", () => {
  it("stores per-user vocabulary and substitutions", async () => {
    const manager = new VocaContextManager(new MemoryVocaContextStore());

    await manager.addVocabularyTerm("alice", "Optagon");
    await manager.addVocabularyTerm("alice", " optagon ");
    await manager.addVocabularyTerm("bob", "Voca");
    await manager.addSubstitution("alice", { from: "super base", to: "Supabase" });

    expect(await manager.toContextOptions("alice")).toEqual({
      context: undefined,
      contextTerms: ["Optagon", "super base", "Supabase"],
    });
    expect(await manager.toContextOptions("bob")).toEqual({
      context: undefined,
      contextTerms: ["Voca"],
    });
  });

  it("lets request context override stored prompt while merging terms", async () => {
    const manager = new VocaContextManager();
    await manager.set({
      userId: "alice",
      context: "Stored prompt",
      vocabulary: ["Optagon"],
      substitutions: [{ from: "get hub", to: "GitHub" }],
    });

    expect(await manager.toContextOptions("alice", {
      context: "Request prompt",
      contextTerms: ["Qwen"],
    })).toEqual({
      context: "Request prompt",
      contextTerms: ["Optagon", "get hub", "GitHub", "Qwen"],
    });
  });

  it("persists per-user context across browser storage backed manager instances", async () => {
    const storage = new TestStorage();
    const firstManager = new VocaContextManager(
      new BrowserLocalStorageVocaContextStore({ storage }),
    );

    await firstManager.set({
      userId: "alice",
      context: "Bias toward the Optagon project.",
      vocabulary: [" Optagon ", "voca", "Optagon"],
      substitutions: [
        { from: "super base", to: "Supabase" },
        { from: "get hub", to: "GitHub" },
      ],
    });
    await firstManager.set({
      userId: "bob",
      vocabulary: ["Frame Manager"],
    });

    const secondManager = new VocaContextManager(
      new BrowserLocalStorageVocaContextStore({ storage }),
    );

    expect(await secondManager.toContextOptions("alice")).toEqual({
      context: "Bias toward the Optagon project.",
      contextTerms: ["Optagon", "voca", "super base", "Supabase", "get hub", "GitHub"],
    });
    expect(await secondManager.toContextOptions("bob")).toEqual({
      context: undefined,
      contextTerms: ["Frame Manager"],
    });
    expect(await secondManager.applySubstitutions("alice", "open super base from get hub")).toBe("open Supabase from GitHub");
  });

  it("supports direct typed profile updates for vocabulary, contacts, and substitutions", async () => {
    const manager = new VocaContextManager();

    await manager.addVocabularyTerms("alice", ["Optagon", "Frame Manager"]);
    await manager.addContacts("alice", [
      { name: "Ada Lovelace", aliases: ["Ada"] },
      { name: "Grace Hopper", aliases: ["Amazing Grace"] },
    ]);
    await manager.addSubstitution("alice", { from: "super base", to: "Supabase" });

    expect(await manager.toContextOptions("alice")).toEqual({
      context: undefined,
      contextTerms: [
        "Optagon",
        "Frame Manager",
        "Ada Lovelace",
        "Ada",
        "Grace Hopper",
        "Amazing Grace",
        "super base",
        "Supabase",
      ],
    });

    await manager.removeContact("alice", "Ada Lovelace");
    expect(await manager.toContextOptions("alice")).toEqual({
      context: undefined,
      contextTerms: ["Optagon", "Frame Manager", "Grace Hopper", "Amazing Grace", "super base", "Supabase"],
    });
  });

  it("applies parsed voice commands to the caller-provided user profile", async () => {
    const manager = new VocaContextManager();

    const contextResult = await manager.handleContextCommand("alice", "voca add context", {
      selectedText: "Optagon Frame",
    });
    const contactResult = await manager.handleContextCommand("alice", "voca add contact Ada Lovelace");
    const substitutionResult = await manager.handleContextCommand("bob", "voca replace get hub with GitHub");

    expect(contextResult.handled).toBe(true);
    expect(contactResult.handled).toBe(true);
    expect(substitutionResult.handled).toBe(true);
    expect(await manager.toContextOptions("alice")).toEqual({
      context: undefined,
      contextTerms: ["Optagon Frame", "Ada Lovelace"],
    });
    expect(await manager.toContextOptions("bob")).toEqual({
      context: undefined,
      contextTerms: ["get hub", "GitHub"],
    });
    expect(await manager.handleContextCommand("alice", "add context", {
      selectedText: "ignored",
    })).toEqual({ handled: false });
  });
});

describe("AuthenticatedVocaContextStore", () => {
  it("uses server identity instead of sending browser userId", async () => {
    const requests: Array<{ method: string; body?: unknown; auth?: string | null }> = [];
    const store = new AuthenticatedVocaContextStore({
      endpoint: "/api/voca/context",
      headers: { Authorization: "Bearer token" },
      fetchImpl: (async (_url, init) => {
        requests.push({
          method: init?.method ?? "GET",
          body: init?.body ? JSON.parse(String(init.body)) : undefined,
          auth: new Headers(init?.headers).get("authorization"),
        });
        return Response.json({
          userId: "server-user",
          context: "Shared frame context.",
        });
      }) as typeof fetch,
    });

    await store.set({
      userId: "browser-controlled-user",
      context: "Shared frame context.",
    });
    const context = await store.get("browser-controlled-user");

    expect(requests[0]).toEqual({
      method: "PUT",
      body: {
        context: "Shared frame context.",
        vocabulary: [],
        contacts: [],
        substitutions: [],
      },
      auth: "Bearer token",
    });
    expect(context?.userId).toBe("server-user");
  });
});

describe("VocaClient", () => {
  it("posts context terms as multipart form data", async () => {
    let capturedUrl = "";
    let capturedForm: FormData | null = null;
    const client = new VocaClient({
      baseUrl: "http://asr.example.test/",
      fetchImpl: (async (url, init) => {
        capturedUrl = String(url);
        capturedForm = init?.body as FormData;
        return Response.json({ text: "hello" });
      }) as typeof fetch,
    });

    const result = await client.transcribe(new Uint8Array([1, 2, 3]), {
      language: "en",
      contextTerms: ["Optagon", "Voca"],
    });

    expect(result.text).toBe("hello");
    expect(capturedUrl).toBe("http://asr.example.test/v1/transcribe?language=en");
    expect(capturedForm).not.toBeNull();
    const form = capturedForm as unknown as FormData;
    expect(form.get("context_terms")).toBe(JSON.stringify(["Optagon", "Voca"]));
  });

  it("transcribes with stored user context and applies output substitutions", async () => {
    let capturedForm: FormData | null = null;
    const contextManager = new VocaContextManager();
    await contextManager.set({
      userId: "alice",
      vocabulary: ["Optagon"],
      substitutions: [{ from: "super base", to: "Supabase" }],
    });
    const client = new VocaClient({
      contextManager,
      fetchImpl: (async (_url, init) => {
        capturedForm = init?.body as FormData;
        return Response.json({ text: "open super base" });
      }) as typeof fetch,
    });

    const result = await client.transcribeForUser("alice", new Uint8Array([1, 2, 3]), {
      contextTerms: ["Qwen"],
    });

    expect(result.text).toBe("open Supabase");
    const form = capturedForm as unknown as FormData;
    expect(form.get("context_terms")).toBe(JSON.stringify(["Optagon", "super base", "Supabase", "Qwen"]));
  });

  it("honors the configured context sharing policy", async () => {
    let capturedForm: FormData | null = null;
    const contextManager = new VocaContextManager();
    await contextManager.set({
      userId: "alice",
      context: "Do not send this prompt remotely.",
      vocabulary: ["Optagon"],
      contacts: [{ name: "Ada Lovelace", aliases: ["Ada"] }],
      substitutions: [{ from: "super base", to: "Supabase" }],
    });
    const client = new VocaClient({
      contextManager,
      contextSharingPolicy: "vocabulary_only",
      fetchImpl: (async (_url, init) => {
        capturedForm = init?.body as FormData;
        return Response.json({ text: "open super base" });
      }) as typeof fetch,
    });

    await client.transcribeForUser("alice", new Uint8Array([1, 2, 3]));

    const form = capturedForm as unknown as FormData;
    expect(form.get("context")).toBeNull();
    expect(form.get("context_terms")).toBe(JSON.stringify(["Optagon", "super base", "Supabase"]));
  });

  it("round-trips batch transcription through a Voca-compatible HTTP endpoint", async () => {
    type ReceivedTranscriptionRequest = {
      auth: string | null;
      language: string | null;
      context: FormDataEntryValue | null;
      contextTerms: unknown;
      fileName: string;
      fileType: string;
      fileSize: number;
    };
    const received: { current: ReceivedTranscriptionRequest | null } = { current: null };

    const server = Bun.serve({
      port: 0,
      async fetch(request) {
        const url = new URL(request.url);
        if (url.pathname !== "/v1/transcribe") return new Response("not found", { status: 404 });

        const form = await request.formData();
        const file = form.get("file");
        if (!(file instanceof File)) return Response.json({ error: "missing file" }, { status: 400 });
        received.current = {
          auth: request.headers.get("authorization"),
          language: url.searchParams.get("language"),
          context: form.get("context"),
          contextTerms: JSON.parse(String(form.get("context_terms") ?? "[]")),
          fileName: file.name,
          fileType: file.type,
          fileSize: file.size,
        };
        return Response.json({ text: "open super base from get hub", processing_time_seconds: 0.01 });
      },
    });

    try {
      const contextManager = new VocaContextManager();
      await contextManager.set({
        userId: "alice",
        context: "Prefer Optagon product names.",
        vocabulary: ["Optagon", "Voca"],
        substitutions: [
          { from: "super base", to: "Supabase" },
          { from: "get hub", to: "GitHub" },
        ],
      });
      const client = new VocaClient({
        baseUrl: `http://127.0.0.1:${server.port}/`,
        authToken: "secret-token",
        contextManager,
      });

      const result = await client.transcribeForUser("alice", new Uint8Array([1, 2, 3, 4]), {
        language: "en",
        filename: "sample.wav",
        mimeType: "audio/wav",
        contextTerms: ["Qwen"],
      });

      expect(result.text).toBe("open Supabase from GitHub");
      expect(received.current).not.toBeNull();
      const seen = received.current as ReceivedTranscriptionRequest;
      expect(seen).toMatchObject({
        auth: "Bearer secret-token",
        language: "en",
        context: "Prefer Optagon product names.",
        contextTerms: ["Optagon", "Voca", "super base", "Supabase", "get hub", "GitHub", "Qwen"],
        fileName: "sample.wav",
        fileSize: 4,
      });
      expect(seen.fileType.startsWith("audio/")).toBe(true);
    } finally {
      server.stop(true);
    }
  });

  it("sends stored user context in the first streaming WebSocket config frame", async () => {
    let resolveConfig: (config: Record<string, unknown>) => void;
    const configReceived = new Promise<Record<string, unknown>>((resolve) => {
      resolveConfig = resolve;
    });
    let resolveFinal: (message: Record<string, unknown>) => void;
    const finalReceived = new Promise<Record<string, unknown>>((resolve) => {
      resolveFinal = resolve;
    });

    const server = Bun.serve({
      port: 0,
      fetch(request, server) {
        const url = new URL(request.url);
        if (url.pathname === "/v1/stream" && server.upgrade(request)) return;
        return new Response("not found", { status: 404 });
      },
      websocket: {
        message(ws, message) {
          if (typeof message !== "string") return;
          const parsed = JSON.parse(message) as Record<string, unknown>;
          if (parsed.action === "finish") {
            ws.send(JSON.stringify({ text: "done with super base", final: true }));
            return;
          }
          resolveConfig(parsed);
          ws.send(JSON.stringify({ status: "streaming", session_id: "session-1" }));
        },
      },
    });

    let stream: ReturnType<VocaClient["stream"]> | null = null;
    try {
      const contextManager = new VocaContextManager();
      await contextManager.set({
        userId: "alice",
        vocabulary: ["Optagon"],
        substitutions: [{ from: "super base", to: "Supabase" }],
      });
      const client = new VocaClient({
        baseUrl: `http://127.0.0.1:${server.port}`,
        authToken: "stream-token",
        contextManager,
      });

      stream = await client.streamForUser("alice", {
        language: "auto",
        contextTerms: ["Qwen"],
      });
      stream.onMessage((message) => {
        if ("text" in message && message.final === true) resolveFinal(message);
      });

      expect(await configReceived).toEqual({
        language: "auto",
        token: "stream-token",
        context_terms: ["Optagon", "super base", "Supabase", "Qwen"],
      });

      stream.finish();
      expect(await finalReceived).toEqual({ text: "done with Supabase", final: true });
    } finally {
      stream?.close();
      server.stop(true);
    }
  });
});
