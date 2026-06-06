export type AsrLanguage = string | "auto";

export interface VocaContextOptions {
  context?: string;
  contextTerms?: string[];
}

export interface VocaSubstitution {
  from: string;
  to: string;
}

export interface VocaContact {
  name: string;
  aliases?: string[];
  notes?: string;
}

export interface VocaUserContext {
  userId: string;
  context?: string;
  vocabulary?: string[];
  contacts?: VocaContact[];
  substitutions?: VocaSubstitution[];
  updatedAt?: string;
}

export interface VocaContextLimits {
  maxContextChars: number;
  maxContextWords: number;
  maxVocabularyTerms: number;
  maxContacts: number;
  maxAliasesPerContact: number;
  maxSubstitutions: number;
  maxTermChars: number;
  maxNotesChars: number;
}

export interface VocaContextStore {
  get(userId: string): Promise<VocaUserContext | null>;
  set(context: VocaUserContext): Promise<void>;
  delete(userId: string): Promise<void>;
  listUserIds?(): Promise<string[]>;
}

export interface VocaClientOptions {
  baseUrl?: string;
  wsUrl?: string;
  authToken?: string;
  fetchImpl?: typeof fetch;
  websocketImpl?: typeof WebSocket;
  contextManager?: VocaContextManager;
  contextSharingPolicy?: VocaContextSharingPolicy;
}

export interface VocaStatus {
  status: "ready" | "not_configured" | string;
  model?: string;
  streaming_supported?: boolean;
  backend?: string;
  backend_preference?: string;
  vllm_error?: string | null;
  kv_cache_mb?: number;
  active_streams?: number;
  max_streams?: number;
  sample_rate?: number;
}

export interface TranscribeOptions extends VocaContextOptions {
  language?: AsrLanguage;
  filename?: string;
  mimeType?: string;
}

export interface TranscribeResult {
  text: string;
  processing_time_seconds?: number;
}

export interface StreamOptions extends VocaContextOptions {
  language?: AsrLanguage;
  token?: string;
}

export type VocaStreamMessage =
  | { status: "streaming"; session_id: string }
  | { partial: string }
  | { text: string; final: true }
  | { error: string; final?: boolean }
  | Record<string, unknown>;

export type VocaStreamMessageTransform = (message: VocaStreamMessage) => VocaStreamMessage | Promise<VocaStreamMessage>;
export type VocaContextSharingPolicy = "none" | "vocabulary_only" | "full";

export type VocaContextCommand =
  | { type: "add_vocabulary"; term: string }
  | { type: "remove_vocabulary"; term: string }
  | { type: "add_contact"; contact: VocaContact }
  | { type: "remove_contact"; name: string }
  | { type: "add_substitution"; from: string; to: string }
  | { type: "remove_substitution"; from: string };

export interface VocaCommandParseOptions {
  selectedText?: string;
}

export interface VocaCommandResult {
  handled: boolean;
  command?: VocaContextCommand;
  context?: VocaUserContext;
  message?: string;
}

export const DEFAULT_VOCA_CONTEXT_LIMITS: VocaContextLimits = {
  maxContextChars: 512,
  maxContextWords: 40,
  maxVocabularyTerms: 100,
  maxContacts: 100,
  maxAliasesPerContact: 10,
  maxSubstitutions: 100,
  maxTermChars: 120,
  maxNotesChars: 240,
};

function countWords(value: string): number {
  return value.trim().split(/\s+/).filter(Boolean).length;
}

function assertNoControlCharacters(value: string, label: string): void {
  if (/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/.test(value)) {
    throw new Error(`${label} contains unsupported control characters.`);
  }
}

function assertMaxLength(value: string, max: number, label: string): void {
  if (value.length > max) {
    throw new Error(`${label} exceeds ${max} characters.`);
  }
}

export function normalizeContextTerms(terms: Iterable<unknown> | null | undefined): string[] {
  if (!terms) return [];
  const seen = new Set<string>();
  const normalized: string[] = [];
  for (const term of terms) {
    const value = String(term).trim();
    if (!value || seen.has(value)) continue;
    seen.add(value);
    normalized.push(value);
  }
  return normalized;
}

export function buildContextPrompt(terms: Iterable<unknown> | null | undefined): string | undefined {
  const normalized = normalizeContextTerms(terms);
  if (normalized.length === 0) return undefined;
  return `The following terms may appear in the audio: ${normalized.join(", ")}.`;
}

function trimTrailingSlash(url: string): string {
  return url.endsWith("/") ? url.slice(0, -1) : url;
}

function defaultWsUrl(baseUrl: string): string {
  return `${trimTrailingSlash(baseUrl).replace(/^http/, "ws")}/v1/stream`;
}

function appendLanguage(url: string, language: AsrLanguage | undefined): string {
  if (!language || language === "auto") return url;
  const parsed = new URL(url);
  parsed.searchParams.set("language", language);
  return parsed.toString();
}

function toBlobPart(audio: ArrayBuffer | Uint8Array): BlobPart {
  if (audio instanceof ArrayBuffer) return audio;
  const bytes = new Uint8Array(audio.byteLength);
  bytes.set(audio);
  return bytes.buffer;
}

function normalizeSubstitution(record: VocaSubstitution): VocaSubstitution | null {
  const from = String(record.from).trim();
  const to = String(record.to).trim();
  if (!from || !to) return null;
  return { from, to };
}

function normalizeContact(record: VocaContact): VocaContact | null {
  const name = String(record.name).trim();
  if (!name) return null;
  return {
    name,
    aliases: normalizeContextTerms(record.aliases),
    notes: record.notes?.trim() || undefined,
  };
}

function contactTerms(contact: VocaContact): string[] {
  return normalizeContextTerms([contact.name, ...(contact.aliases ?? [])]);
}

function stripCommandPunctuation(text: string): string {
  return text.trim().replace(/[.!?]+$/g, "").trim();
}

function selectedOrInlineValue(inline: string | undefined, options: VocaCommandParseOptions): string | null {
  const selected = options.selectedText?.trim();
  if (selected) return selected;
  const value = inline?.trim();
  return value || null;
}

function splitReplacement(value: string): VocaSubstitution | null {
  const match = value.match(/^(.+?)\s+(?:with|to|as)\s+(.+)$/i);
  if (!match) return null;
  return normalizeSubstitution({ from: match[1], to: match[2] });
}

function uniqueByLowercase(values: string[]): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const value of values) {
    const key = value.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(value);
  }
  return result;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function mergeContextOptions(base: VocaContextOptions, overlay: VocaContextOptions): VocaContextOptions {
  return {
    ...base,
    context: overlay.context ?? base.context,
    contextTerms: uniqueByLowercase([
      ...normalizeContextTerms(base.contextTerms),
      ...normalizeContextTerms(overlay.contextTerms),
    ]),
  };
}

export class MemoryVocaContextStore implements VocaContextStore {
  private readonly contexts = new Map<string, VocaUserContext>();

  async get(userId: string): Promise<VocaUserContext | null> {
    const context = this.contexts.get(userId);
    return context ? structuredClone(context) : null;
  }

  async set(context: VocaUserContext): Promise<void> {
    this.contexts.set(context.userId, structuredClone(context));
  }

  async delete(userId: string): Promise<void> {
    this.contexts.delete(userId);
  }

  async listUserIds(): Promise<string[]> {
    return [...this.contexts.keys()].sort();
  }
}

export class BrowserLocalStorageVocaContextStore implements VocaContextStore {
  private readonly prefix: string;
  private readonly storage: Storage;

  constructor(options: { prefix?: string; storage?: Storage } = {}) {
    if (!options.storage && typeof localStorage === "undefined") {
      throw new Error("BrowserLocalStorageVocaContextStore requires localStorage or an explicit storage implementation.");
    }
    this.prefix = options.prefix ?? "voca:context:";
    this.storage = options.storage ?? localStorage;
  }

  async get(userId: string): Promise<VocaUserContext | null> {
    const raw = this.storage.getItem(this.key(userId));
    if (!raw) return null;
    return normalizeUserContext(JSON.parse(raw) as VocaUserContext);
  }

  async set(context: VocaUserContext): Promise<void> {
    this.storage.setItem(this.key(context.userId), JSON.stringify(normalizeUserContext(context)));
  }

  async delete(userId: string): Promise<void> {
    this.storage.removeItem(this.key(userId));
  }

  async listUserIds(): Promise<string[]> {
    const ids: string[] = [];
    for (let i = 0; i < this.storage.length; i++) {
      const key = this.storage.key(i);
      if (key?.startsWith(this.prefix)) ids.push(key.slice(this.prefix.length));
    }
    return ids.sort();
  }

  private key(userId: string): string {
    return `${this.prefix}${userId}`;
  }
}

export function normalizeUserContext(context: VocaUserContext): VocaUserContext {
  const substitutions = (context.substitutions ?? [])
    .map(normalizeSubstitution)
    .filter((record): record is VocaSubstitution => record !== null);
  const contacts = (context.contacts ?? [])
    .map(normalizeContact)
    .filter((record): record is VocaContact => record !== null);

  return {
    userId: String(context.userId).trim(),
    context: context.context?.trim() || undefined,
    vocabulary: normalizeContextTerms(context.vocabulary),
    contacts,
    substitutions,
    updatedAt: context.updatedAt,
  };
}

export function validateVocaUserContext(
  context: VocaUserContext,
  limits: VocaContextLimits = DEFAULT_VOCA_CONTEXT_LIMITS,
): VocaUserContext {
  const normalized = normalizeUserContext(context);
  assertNoControlCharacters(normalized.userId, "userId");
  if (!normalized.userId) throw new Error("Voca user context requires a userId.");

  if (normalized.context) {
    assertNoControlCharacters(normalized.context, "context");
    assertMaxLength(normalized.context, limits.maxContextChars, "context");
    if (countWords(normalized.context) > limits.maxContextWords) {
      throw new Error(`context exceeds ${limits.maxContextWords} words.`);
    }
  }

  if ((normalized.vocabulary?.length ?? 0) > limits.maxVocabularyTerms) {
    throw new Error(`vocabulary exceeds ${limits.maxVocabularyTerms} terms.`);
  }
  for (const term of normalized.vocabulary ?? []) {
    assertNoControlCharacters(term, "vocabulary term");
    assertMaxLength(term, limits.maxTermChars, "vocabulary term");
  }

  if ((normalized.contacts?.length ?? 0) > limits.maxContacts) {
    throw new Error(`contacts exceeds ${limits.maxContacts} entries.`);
  }
  for (const contact of normalized.contacts ?? []) {
    assertNoControlCharacters(contact.name, "contact name");
    assertMaxLength(contact.name, limits.maxTermChars, "contact name");
    if ((contact.aliases?.length ?? 0) > limits.maxAliasesPerContact) {
      throw new Error(`contact aliases exceeds ${limits.maxAliasesPerContact} entries.`);
    }
    for (const alias of contact.aliases ?? []) {
      assertNoControlCharacters(alias, "contact alias");
      assertMaxLength(alias, limits.maxTermChars, "contact alias");
    }
    if (contact.notes) {
      assertNoControlCharacters(contact.notes, "contact notes");
      assertMaxLength(contact.notes, limits.maxNotesChars, "contact notes");
    }
  }

  if ((normalized.substitutions?.length ?? 0) > limits.maxSubstitutions) {
    throw new Error(`substitutions exceeds ${limits.maxSubstitutions} entries.`);
  }
  for (const substitution of normalized.substitutions ?? []) {
    assertNoControlCharacters(substitution.from, "substitution source");
    assertNoControlCharacters(substitution.to, "substitution target");
    assertMaxLength(substitution.from, limits.maxTermChars, "substitution source");
    assertMaxLength(substitution.to, limits.maxTermChars, "substitution target");
  }

  return normalized;
}

export function contextOptionsFromUserContext(
  context: VocaUserContext | null | undefined,
  policy: VocaContextSharingPolicy = "full",
): VocaContextOptions {
  if (!context) return {};
  if (policy === "none") return {};
  const normalized = normalizeUserContext(context);
  const substitutionTerms = normalized.substitutions?.flatMap(({ from, to }) => [from, to]) ?? [];
  const contactVocabulary = normalized.contacts?.flatMap(contactTerms) ?? [];
  if (policy === "vocabulary_only") {
    return {
      contextTerms: uniqueByLowercase([
        ...normalizeContextTerms(normalized.vocabulary),
        ...normalizeContextTerms(substitutionTerms),
      ]),
    };
  }
  return {
    context: normalized.context,
    contextTerms: uniqueByLowercase([
      ...normalizeContextTerms(normalized.vocabulary),
      ...contactVocabulary,
      ...normalizeContextTerms(substitutionTerms),
    ]),
  };
}

export function applyVocaSubstitutions(text: string, substitutions: Iterable<VocaSubstitution> | null | undefined): string {
  const records = [...(substitutions ?? [])]
    .map(normalizeSubstitution)
    .filter((record): record is VocaSubstitution => record !== null)
    .sort((a, b) => b.from.length - a.from.length);
  if (records.length === 0 || !text) return text;

  const lookup = new Map(records.map((record) => [record.from.toLowerCase(), record.to]));
  const pattern = new RegExp(`\\b(${records.map((record) => escapeRegExp(record.from)).join("|")})\\b`, "gi");
  return text.replace(pattern, (match) => lookup.get(match.toLowerCase()) ?? match);
}

export function parseVocaContextCommand(transcript: string, options: VocaCommandParseOptions = {}): VocaContextCommand | null {
  const text = stripCommandPunctuation(transcript);
  const normalized = text.replace(/\s+/g, " ");
  const prefix = normalized.match(/^(?:voice\s*box|voca)\s+(.+)$/i);
  if (!prefix) return null;

  const body = prefix[1].trim();
  const addContext = body.match(/^(?:add|save|remember)\s+(?:this\s+)?(?:context|word|term|phrase|vocabulary)(?:\s+(.+))?$/i);
  if (addContext) {
    const term = selectedOrInlineValue(addContext[1], options);
    return term ? { type: "add_vocabulary", term } : null;
  }

  const removeContext = body.match(/^(?:remove|forget|delete)\s+(?:this\s+)?(?:context|word|term|phrase|vocabulary)(?:\s+(.+))?$/i);
  if (removeContext) {
    const term = selectedOrInlineValue(removeContext[1], options);
    return term ? { type: "remove_vocabulary", term } : null;
  }

  const addContact = body.match(/^(?:add|save|remember)\s+(?:this\s+)?contact(?:\s+(.+))?$/i);
  if (addContact) {
    const name = selectedOrInlineValue(addContact[1], options);
    return name ? { type: "add_contact", contact: { name } } : null;
  }

  const removeContact = body.match(/^(?:remove|forget|delete)\s+(?:this\s+)?contact(?:\s+(.+))?$/i);
  if (removeContact) {
    const name = selectedOrInlineValue(removeContact[1], options);
    return name ? { type: "remove_contact", name } : null;
  }

  const replacement = body.match(/^(?:replace|substitute|correct)\s+(.+)$/i);
  if (replacement) {
    const substitution = splitReplacement(replacement[1]);
    return substitution ? { type: "add_substitution", ...substitution } : null;
  }

  const removeReplacement = body.match(/^(?:remove|forget|delete)\s+(?:replacement|substitution|correction)(?:\s+(.+))?$/i);
  if (removeReplacement) {
    const from = selectedOrInlineValue(removeReplacement[1], options);
    return from ? { type: "remove_substitution", from } : null;
  }

  return null;
}

export class VocaContextManager {
  readonly store: VocaContextStore;

  constructor(store: VocaContextStore = new MemoryVocaContextStore()) {
    this.store = store;
  }

  async get(userId: string): Promise<VocaUserContext> {
    const existing = await this.store.get(userId);
    return existing ?? this.empty(userId);
  }

  async set(context: VocaUserContext): Promise<VocaUserContext> {
    const normalized = validateVocaUserContext({
      ...context,
      updatedAt: new Date().toISOString(),
    });
    await this.store.set(normalized);
    return normalized;
  }

  async update(userId: string, patch: Partial<Omit<VocaUserContext, "userId">>): Promise<VocaUserContext> {
    const current = await this.get(userId);
    return this.set({ ...current, ...patch, userId });
  }

  async addVocabularyTerm(userId: string, term: string): Promise<VocaUserContext> {
    const current = await this.get(userId);
    return this.set({
      ...current,
      vocabulary: uniqueByLowercase([...normalizeContextTerms(current.vocabulary), ...normalizeContextTerms([term])]),
    });
  }

  async addVocabularyTerms(userId: string, terms: Iterable<unknown>): Promise<VocaUserContext> {
    const current = await this.get(userId);
    return this.set({
      ...current,
      vocabulary: uniqueByLowercase([...normalizeContextTerms(current.vocabulary), ...normalizeContextTerms(terms)]),
    });
  }

  async removeVocabularyTerm(userId: string, term: string): Promise<VocaUserContext> {
    const termKey = term.trim().toLowerCase();
    const current = await this.get(userId);
    return this.set({
      ...current,
      vocabulary: normalizeContextTerms(current.vocabulary).filter((value) => value.toLowerCase() !== termKey),
    });
  }

  async addContact(userId: string, contact: VocaContact): Promise<VocaUserContext> {
    const normalized = normalizeContact(contact);
    if (!normalized) return this.get(userId);

    const current = await this.get(userId);
    const others = (current.contacts ?? []).filter(
      (record) => record.name.trim().toLowerCase() !== normalized.name.toLowerCase(),
    );
    return this.set({ ...current, contacts: [...others, normalized] });
  }

  async addContacts(userId: string, contacts: Iterable<VocaContact>): Promise<VocaUserContext> {
    let context = await this.get(userId);
    for (const contact of contacts) {
      const normalized = normalizeContact(contact);
      if (!normalized) continue;
      const others = (context.contacts ?? []).filter(
        (record) => record.name.trim().toLowerCase() !== normalized.name.toLowerCase(),
      );
      context = { ...context, contacts: [...others, normalized] };
    }
    return this.set(context);
  }

  async removeContact(userId: string, name: string): Promise<VocaUserContext> {
    const key = name.trim().toLowerCase();
    const current = await this.get(userId);
    return this.set({
      ...current,
      contacts: (current.contacts ?? []).filter((record) => record.name.trim().toLowerCase() !== key),
    });
  }

  async addSubstitution(userId: string, substitution: VocaSubstitution): Promise<VocaUserContext> {
    const normalized = normalizeSubstitution(substitution);
    if (!normalized) return this.get(userId);

    const current = await this.get(userId);
    const others = (current.substitutions ?? []).filter(
      (record) => record.from.trim().toLowerCase() !== normalized.from.toLowerCase(),
    );
    return this.set({ ...current, substitutions: [...others, normalized] });
  }

  async removeSubstitution(userId: string, from: string): Promise<VocaUserContext> {
    const key = from.trim().toLowerCase();
    const current = await this.get(userId);
    return this.set({
      ...current,
      substitutions: (current.substitutions ?? []).filter((record) => record.from.trim().toLowerCase() !== key),
    });
  }

  async toContextOptions(
    userId: string,
    options: VocaContextOptions = {},
    policy: VocaContextSharingPolicy = "full",
  ): Promise<VocaContextOptions> {
    return mergeContextOptions(contextOptionsFromUserContext(await this.get(userId), policy), options);
  }

  async applySubstitutions(userId: string, text: string): Promise<string> {
    const context = await this.get(userId);
    return applyVocaSubstitutions(text, context.substitutions);
  }

  async processTranscript(userId: string, text: string): Promise<string> {
    return this.applySubstitutions(userId, text);
  }

  async applyContextCommand(userId: string, command: VocaContextCommand): Promise<VocaUserContext> {
    switch (command.type) {
      case "add_vocabulary":
        return this.addVocabularyTerm(userId, command.term);
      case "remove_vocabulary":
        return this.removeVocabularyTerm(userId, command.term);
      case "add_contact":
        return this.addContact(userId, command.contact);
      case "remove_contact":
        return this.removeContact(userId, command.name);
      case "add_substitution":
        return this.addSubstitution(userId, { from: command.from, to: command.to });
      case "remove_substitution":
        return this.removeSubstitution(userId, command.from);
    }
  }

  async handleContextCommand(userId: string, transcript: string, options: VocaCommandParseOptions = {}): Promise<VocaCommandResult> {
    const command = parseVocaContextCommand(transcript, options);
    if (!command) return { handled: false };
    const context = await this.applyContextCommand(userId, command);
    return {
      handled: true,
      command,
      context,
      message: this.commandMessage(command),
    };
  }

  private commandMessage(command: VocaContextCommand): string {
    switch (command.type) {
      case "add_vocabulary":
        return `Added context term: ${command.term}`;
      case "remove_vocabulary":
        return `Removed context term: ${command.term}`;
      case "add_contact":
        return `Added contact: ${command.contact.name}`;
      case "remove_contact":
        return `Removed contact: ${command.name}`;
      case "add_substitution":
        return `Added substitution: ${command.from} -> ${command.to}`;
      case "remove_substitution":
        return `Removed substitution: ${command.from}`;
    }
  }

  private empty(userId: string): VocaUserContext {
    return { userId, vocabulary: [], contacts: [], substitutions: [] };
  }
}

export type VocaContextHeaders =
  | HeadersInit
  | (() => HeadersInit | Promise<HeadersInit>);

export interface AuthenticatedVocaContextStoreOptions {
  endpoint?: string;
  fetchImpl?: typeof fetch;
  headers?: VocaContextHeaders;
}

export class AuthenticatedVocaContextStore implements VocaContextStore {
  private readonly endpoint: string;
  private readonly fetchImpl: typeof fetch;
  private readonly headers?: VocaContextHeaders;

  constructor(options: AuthenticatedVocaContextStoreOptions = {}) {
    this.endpoint = options.endpoint ?? "/api/voca/context";
    this.fetchImpl = options.fetchImpl ?? fetch;
    this.headers = options.headers;
  }

  async get(_userId: string): Promise<VocaUserContext | null> {
    const response = await this.fetchImpl(this.endpoint, {
      method: "GET",
      headers: await this.resolveHeaders(),
    });
    if (response.status === 404) return null;
    if (!response.ok) throw new Error(`Voca context fetch failed: HTTP ${response.status}`);
    return validateVocaUserContext(await response.json() as VocaUserContext);
  }

  async set(context: VocaUserContext): Promise<void> {
    const normalized = validateVocaUserContext(context);
    const { userId: _userId, ...payload } = normalized;
    const response = await this.fetchImpl(this.endpoint, {
      method: "PUT",
      headers: await this.resolveHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(`Voca context save failed: HTTP ${response.status}`);
  }

  async delete(_userId: string): Promise<void> {
    const response = await this.fetchImpl(this.endpoint, {
      method: "DELETE",
      headers: await this.resolveHeaders(),
    });
    if (!response.ok && response.status !== 404) {
      throw new Error(`Voca context delete failed: HTTP ${response.status}`);
    }
  }

  private async resolveHeaders(extra: HeadersInit = {}): Promise<HeadersInit> {
    const base = typeof this.headers === "function" ? await this.headers() : this.headers;
    return { ...(base ?? {}), ...extra };
  }
}

export class VocaStream {
  readonly socket: WebSocket;
  private readonly messageTransform?: VocaStreamMessageTransform;

  constructor(socket: WebSocket, options: StreamOptions = {}, messageTransform?: VocaStreamMessageTransform) {
    this.socket = socket;
    this.messageTransform = messageTransform;
    this.socket.addEventListener("open", () => {
      this.socket.send(JSON.stringify(streamConfig(options)));
    }, { once: true });
  }

  sendPcm16(audio: ArrayBuffer | Uint8Array): void {
    this.socket.send(audio);
  }

  finish(): void {
    this.socket.send(JSON.stringify({ action: "finish" }));
  }

  close(code?: number, reason?: string): void {
    this.socket.close(code, reason);
  }

  onMessage(listener: (message: VocaStreamMessage) => void): void {
    this.socket.addEventListener("message", async (event) => {
      if (typeof event.data !== "string") return;
      const message = JSON.parse(event.data) as VocaStreamMessage;
      listener(this.messageTransform ? await this.messageTransform(message) : message);
    });
  }
}

export class VocaClient {
  private readonly baseUrl: string;
  private readonly wsUrl: string;
  private readonly authToken?: string;
  private readonly fetchImpl: typeof fetch;
  private readonly websocketImpl: typeof WebSocket;
  private readonly contextSharingPolicy: VocaContextSharingPolicy;
  readonly contextManager?: VocaContextManager;

  constructor(options: VocaClientOptions = {}) {
    this.baseUrl = trimTrailingSlash(options.baseUrl ?? "http://127.0.0.1:9876");
    this.wsUrl = options.wsUrl ?? defaultWsUrl(this.baseUrl);
    this.authToken = options.authToken;
    this.fetchImpl = options.fetchImpl ?? fetch;
    this.websocketImpl = options.websocketImpl ?? WebSocket;
    this.contextSharingPolicy = options.contextSharingPolicy ?? "full";
    this.contextManager = options.contextManager;
  }

  async status(): Promise<VocaStatus> {
    const response = await this.fetchImpl(`${this.baseUrl}/v1/status`);
    if (!response.ok) throw new Error(`ASR status failed: HTTP ${response.status}`);
    return await response.json() as VocaStatus;
  }

  async transcribe(audio: Blob | ArrayBuffer | Uint8Array, options: TranscribeOptions = {}): Promise<TranscribeResult> {
    const form = new FormData();
    const file = audio instanceof Blob
      ? audio
      : new Blob([toBlobPart(audio)], { type: options.mimeType ?? "audio/wav" });
    form.set("file", file, options.filename ?? "audio.wav");
    if (options.context) form.set("context", options.context);
    if (options.contextTerms?.length) form.set("context_terms", JSON.stringify(normalizeContextTerms(options.contextTerms)));

    const response = await this.fetchImpl(appendLanguage(`${this.baseUrl}/v1/transcribe`, options.language), {
      method: "POST",
      headers: this.authToken ? { Authorization: `Bearer ${this.authToken}` } : undefined,
      body: form,
    });
    if (!response.ok) throw new Error(`ASR transcription failed: HTTP ${response.status}`);
    return await response.json() as TranscribeResult;
  }

  stream(options: StreamOptions = {}): VocaStream {
    return new VocaStream(new this.websocketImpl(this.wsUrl), {
      ...options,
      token: options.token ?? this.authToken,
    });
  }

  async transcribeForUser(userId: string, audio: Blob | ArrayBuffer | Uint8Array, options: TranscribeOptions = {}): Promise<TranscribeResult> {
    if (!this.contextManager) throw new Error("transcribeForUser requires a VocaContextManager.");
    const userOptions = await this.contextManager.toContextOptions(userId, options, this.contextSharingPolicy);
    const result = await this.transcribe(audio, { ...options, ...userOptions });
    return {
      ...result,
      text: await this.contextManager.processTranscript(userId, result.text),
    };
  }

  async streamForUser(userId: string, options: StreamOptions = {}): Promise<VocaStream> {
    if (!this.contextManager) throw new Error("streamForUser requires a VocaContextManager.");
    const userOptions = await this.contextManager.toContextOptions(userId, options, this.contextSharingPolicy);
    return new VocaStream(new this.websocketImpl(this.wsUrl), {
      ...options,
      ...userOptions,
      token: options.token ?? this.authToken,
    }, async (message) => {
      if ("text" in message && typeof message.text === "string" && "final" in message && message.final === true) {
        return {
          ...message,
          text: await this.contextManager!.processTranscript(userId, message.text),
        };
      }
      return message;
    });
  }
}

export function streamConfig(options: StreamOptions = {}): Record<string, unknown> {
  const config: Record<string, unknown> = {};
  if (options.language) config.language = options.language;
  if (options.token) config.token = options.token;
  if (options.context) config.context = options.context;
  if (options.contextTerms?.length) config.context_terms = normalizeContextTerms(options.contextTerms);
  return config;
}

/** @deprecated Use VocaContextOptions. */
export type VoiceboxContextOptions = VocaContextOptions;
/** @deprecated Use VocaSubstitution. */
export type VoiceboxSubstitution = VocaSubstitution;
/** @deprecated Use VocaContact. */
export type VoiceboxContact = VocaContact;
/** @deprecated Use VocaUserContext. */
export type VoiceboxUserContext = VocaUserContext;
/** @deprecated Use VocaContextLimits. */
export type VoiceboxContextLimits = VocaContextLimits;
/** @deprecated Use VocaContextStore. */
export type VoiceboxContextStore = VocaContextStore;
/** @deprecated Use VocaClientOptions. */
export type VoiceboxClientOptions = VocaClientOptions;
/** @deprecated Use VocaStatus. */
export type VoiceboxStatus = VocaStatus;
/** @deprecated Use VocaStreamMessage. */
export type VoiceboxStreamMessage = VocaStreamMessage;
/** @deprecated Use VocaStreamMessageTransform. */
export type VoiceboxStreamMessageTransform = VocaStreamMessageTransform;
/** @deprecated Use VocaContextSharingPolicy. */
export type VoiceboxContextSharingPolicy = VocaContextSharingPolicy;
/** @deprecated Use VocaContextCommand. */
export type VoiceboxContextCommand = VocaContextCommand;
/** @deprecated Use VocaCommandParseOptions. */
export type VoiceboxCommandParseOptions = VocaCommandParseOptions;
/** @deprecated Use VocaCommandResult. */
export type VoiceboxCommandResult = VocaCommandResult;

/** @deprecated Use DEFAULT_VOCA_CONTEXT_LIMITS. */
export const DEFAULT_VOICEBOX_CONTEXT_LIMITS = DEFAULT_VOCA_CONTEXT_LIMITS;
/** @deprecated Use MemoryVocaContextStore. */
export const MemoryVoiceboxContextStore = MemoryVocaContextStore;
/** @deprecated Use BrowserLocalStorageVocaContextStore. */
export const BrowserLocalStorageVoiceboxContextStore = BrowserLocalStorageVocaContextStore;
/** @deprecated Use validateVocaUserContext. */
export const validateVoiceboxUserContext = validateVocaUserContext;
/** @deprecated Use applyVocaSubstitutions. */
export const applyVoiceboxSubstitutions = applyVocaSubstitutions;
/** @deprecated Use parseVocaContextCommand. */
export const parseVoiceboxContextCommand = parseVocaContextCommand;
/** @deprecated Use VocaContextManager. */
export const VoiceboxContextManager = VocaContextManager;
/** @deprecated Use AuthenticatedVocaContextStore. */
export const AuthenticatedVoiceboxContextStore = AuthenticatedVocaContextStore;
/** @deprecated Use VocaStream. */
export const VoiceboxStream = VocaStream;
/** @deprecated Use VocaClient. */
export const VoiceboxClient = VocaClient;
