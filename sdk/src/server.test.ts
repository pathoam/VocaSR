import { mkdtemp, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { afterEach, describe, expect, it } from "bun:test";
import { FileVocaContextStore, handleVocaContextRequest } from "./server.js";

let tempDirs: string[] = [];

afterEach(async () => {
  await Promise.all(tempDirs.map((dir) => rm(dir, { recursive: true, force: true })));
  tempDirs = [];
});

async function tempDir(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), "voca-context-"));
  tempDirs.push(dir);
  return dir;
}

describe("FileVocaContextStore", () => {
  it("persists context by server-side scope id", async () => {
    const store = new FileVocaContextStore({ rootDir: await tempDir() });

    await store.set({
      userId: "frame_abc__user_alice@example.com",
      context: "Prefer Optagon project terminology.",
      vocabulary: ["Voca"],
    });

    expect(await store.get("frame_abc__user_alice@example.com")).toMatchObject({
      userId: "frame_abc__user_alice@example.com",
      context: "Prefer Optagon project terminology.",
      vocabulary: ["Voca"],
    });
    expect(await store.get("frame_abc__user_bob@example.com")).toBeNull();
  });
});

describe("handleVocaContextRequest", () => {
  it("serves GET PUT DELETE without trusting browser-supplied userId", async () => {
    const store = new FileVocaContextStore({ rootDir: await tempDir() });
    const options = {
      store,
      resolveUserId: () => "server-user",
    };

    const put = await handleVocaContextRequest(new Request("http://test/context", {
      method: "PUT",
      body: JSON.stringify({
        userId: "browser-user",
        context: "Shared context across devices.",
      }),
    }), options);
    expect(put.status).toBe(200);
    expect(await put.json()).toMatchObject({
      userId: "server-user",
      context: "Shared context across devices.",
    });

    const get = await handleVocaContextRequest(new Request("http://test/context"), options);
    expect(await get.json()).toMatchObject({
      userId: "server-user",
      context: "Shared context across devices.",
    });

    const browserUserRecord = await store.get("browser-user");
    expect(browserUserRecord).toBeNull();

    const del = await handleVocaContextRequest(new Request("http://test/context", {
      method: "DELETE",
    }), options);
    expect(del.status).toBe(200);
    expect(await store.get("server-user")).toBeNull();
  });

  it("accepts resolveScopeId as the clearer server-derived identity hook", async () => {
    const store = new FileVocaContextStore({ rootDir: await tempDir() });
    const options = {
      store,
      resolveScopeId: () => "frame_abc__user_alice",
    };

    const put = await handleVocaContextRequest(new Request("http://test/context", {
      method: "PUT",
      body: JSON.stringify({ context: "Frame-scoped personal terms." }),
    }), options);
    expect(put.status).toBe(200);

    expect(await store.get("frame_abc__user_alice")).toMatchObject({
      userId: "frame_abc__user_alice",
      context: "Frame-scoped personal terms.",
    });
  });
});
