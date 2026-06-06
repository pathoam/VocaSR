import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import {
  validateVocaUserContext,
  type VocaContextLimits,
  type VocaContextStore,
  type VocaUserContext,
} from "./index.js";

export interface FileVocaContextStoreOptions {
  rootDir: string;
  limits?: VocaContextLimits;
}

export class FileVocaContextStore implements VocaContextStore {
  private readonly rootDir: string;
  private readonly limits?: VocaContextLimits;

  constructor(options: FileVocaContextStoreOptions) {
    this.rootDir = options.rootDir;
    this.limits = options.limits;
  }

  async get(userId: string): Promise<VocaUserContext | null> {
    try {
      const raw = await readFile(this.pathForUser(userId), "utf8");
      return validateVocaUserContext(JSON.parse(raw) as VocaUserContext, this.limits);
    } catch (error) {
      if (isNotFoundError(error)) return null;
      throw error;
    }
  }

  async set(context: VocaUserContext): Promise<void> {
    const normalized = validateVocaUserContext(context, this.limits);
    const path = this.pathForUser(normalized.userId);
    await mkdir(dirname(path), { recursive: true });
    await writeFile(path, JSON.stringify(normalized, null, 2) + "\n", "utf8");
  }

  async delete(userId: string): Promise<void> {
    await rm(this.pathForUser(userId), { force: true });
  }

  private pathForUser(userId: string): string {
    const safeId = userId.trim().replace(/[^A-Za-z0-9_.-]/g, "_");
    if (!safeId) throw new Error("Voca user context requires a userId.");
    return join(this.rootDir, `${safeId}.json`);
  }
}

export interface VocaContextRequestHandlerOptions {
  store: VocaContextStore;
  resolveUserId?: (request: Request) => string | Promise<string>;
  resolveScopeId?: (request: Request) => string | Promise<string>;
  limits?: VocaContextLimits;
}

export async function handleVocaContextRequest(
  request: Request,
  options: VocaContextRequestHandlerOptions,
): Promise<Response> {
  const resolveScopeId = options.resolveScopeId ?? options.resolveUserId;
  if (!resolveScopeId) return json({ error: "missing_scope_resolver" }, 500);
  const userId = (await resolveScopeId(request)).trim();
  if (!userId) return json({ error: "unauthorized" }, 401);

  if (request.method === "GET") {
    const context = await options.store.get(userId);
    if (!context) return json({ error: "not_found" }, 404);
    return json(context);
  }

  if (request.method === "PUT") {
    const body = await request.json().catch(() => null) as Partial<VocaUserContext> | null;
    if (!body || typeof body !== "object") return json({ error: "invalid_json" }, 400);
    const context = validateVocaUserContext({
      ...body,
      userId,
      updatedAt: new Date().toISOString(),
    }, options.limits);
    await options.store.set(context);
    return json(context);
  }

  if (request.method === "DELETE") {
    await options.store.delete(userId);
    return json({ ok: true });
  }

  return json({ error: "method_not_allowed" }, 405, { Allow: "GET, PUT, DELETE" });
}

function json(body: unknown, status = 200, headers: Record<string, string> = {}): Response {
  return Response.json(body, {
    status,
    headers: { ...headers, "Cache-Control": "no-store" },
  });
}

function isNotFoundError(error: unknown): boolean {
  return typeof error === "object"
    && error !== null
    && "code" in error
    && (error as { code?: string }).code === "ENOENT";
}

/** @deprecated Use FileVocaContextStoreOptions. */
export type FileVoiceboxContextStoreOptions = FileVocaContextStoreOptions;
/** @deprecated Use FileVocaContextStore. */
export const FileVoiceboxContextStore = FileVocaContextStore;
/** @deprecated Use VocaContextRequestHandlerOptions. */
export type VoiceboxContextRequestHandlerOptions = VocaContextRequestHandlerOptions;
/** @deprecated Use handleVocaContextRequest. */
export const handleVoiceboxContextRequest = handleVocaContextRequest;
