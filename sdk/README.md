# @optagon/voca-sdk

Browser and JavaScript client helpers for Optagon Voca-compatible ASR
endpoints.

## Per-user ASR context

Voca accepts request context as `context` and `context_terms`. The SDK keeps
the user-facing model higher level:

- `vocabulary`: words and phrases that should bias ASR decoding.
- `substitutions`: misheard text to corrected text, also added to ASR context.
- `context`: optional free-form prompt sent to Voca.

```ts
import {
  BrowserLocalStorageVocaContextStore,
  VocaClient,
  VocaContextManager,
} from "@optagon/voca-sdk";

const contextManager = new VocaContextManager(
  new BrowserLocalStorageVocaContextStore(),
);

await contextManager.addVocabularyTerm(user.id, "Optagon");
await contextManager.addSubstitution(user.id, {
  from: "super base",
  to: "Supabase",
});
await contextManager.addContact(user.id, {
  name: "Ada Lovelace",
  aliases: ["Ada"],
});

const client = new VocaClient({ contextManager });
const result = await client.transcribeForUser(user.id, audioBlob, {
  language: "en",
});
```

For streaming:

```ts
const stream = await client.streamForUser(user.id, { language: "auto" });
stream.onMessage((message) => {
  // Handle partial/final Voca messages.
});
```

`transcribeForUser()` applies stored substitutions to the final transcript after
Voca returns. `streamForUser()` sends the stored context to Voca and
applies stored substitutions to final stream messages before invoking
`onMessage()` listeners.

## Direct profile updates

Companion should pass its own stable user id into every profile call. The SDK
does not infer identity.

```ts
await contextManager.addVocabularyTerms(user.id, ["Optagon", "Frame Manager"]);
await contextManager.addContacts(user.id, [
  { name: "Ada Lovelace", aliases: ["Ada"] },
  { name: "Grace Hopper" },
]);
await contextManager.addSubstitution(user.id, {
  from: "get hub",
  to: "GitHub",
});
```

Contacts, vocabulary, and substitution source/target phrases are all included in
Voca `context_terms` for future ASR requests.

For remote or managed ASR providers, choose an explicit context sharing policy:

```ts
const client = new VocaClient({
  contextManager,
  contextSharingPolicy: "vocabulary_only", // "none" | "vocabulary_only" | "full"
});
```

`full` sends vocabulary, substitutions, contacts, and free-form context. Use a
more restrictive policy when the ASR backend is outside the user's local trust
boundary.

## Voice commands

The SDK can also intercept final transcripts that start with `voca` or
`voice box` and update the caller-provided user's profile. Companion can provide
selected text for commands like "voca add context".

```ts
const result = await contextManager.handleContextCommand(user.id, finalText, {
  selectedText: getSelectedText(),
});

if (result.handled) {
  // Show result.message if useful, and do not insert the command transcript.
  return;
}

insertIntoComposer(await contextManager.processTranscript(user.id, finalText));
```

Examples:

- `voca add context` with selected text `Frame Manager`
- `voca add contact Ada Lovelace`
- `voca replace super base with Supabase`

## Storage

The SDK includes:

- `MemoryVocaContextStore` for tests, server-side composition, and ephemeral
  sessions.
- `BrowserLocalStorageVocaContextStore` for browser persistence.

`BrowserLocalStorageVocaContextStore` is intended for single-user local apps
and demos. Shared or account-based deployments should use the SDK's
server-backed store helpers so the authenticated server decides which scope can
be read or written; do not trust a browser-supplied `userId` as an authorization
boundary. A scope can be a user id, but in Optagon it is usually a composite
frame/user key.

Implement `VocaContextStore` to persist context in Optagon account storage,
SQLite, IndexedDB, or another per-user backend.

Browser client:

```ts
const contextManager = new VocaContextManager(
  new AuthenticatedVocaContextStore({
    endpoint: "/api/voca/context",
    headers: () => ({ Authorization: `Bearer ${getToken()}` }),
  }),
);
```

Server host:

```ts
import {
  FileVocaContextStore,
  handleVocaContextRequest,
} from "@optagon/voca-sdk/server";

const store = new FileVocaContextStore({
  rootDir: "/opt/frame-data/voca-context",
});

const response = await handleVocaContextRequest(request, {
  store,
  resolveScopeId: (request) => {
    const frameId = getAuthenticatedFrameId(request);
    const userId = getAuthenticatedUserId(request);
    return `frame_${frameId}__user_${userId}`;
  },
});
```

The server handler overwrites any browser-provided `userId` with the
server-derived scope id and enforces small context limits before saving.
`resolveUserId` remains supported as a compatibility alias for older hosts.
