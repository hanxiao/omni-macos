# Linked Macs: federated search between copies of Omni (2026-10-12)

The plan as agreed with Han, then the findings as they come. A Mac running Omni (A) searches
another Mac running Omni (B) with B's index and B's GPU, and shows B's hits in A's window as if they
were a folder. No server of ours anywhere: discovery is Bonjour on the local network, or an address
the user types (a VPN they already run: Tailscale, WireGuard, an office VPN).

## Decisions (Han, 2026-10-12)

- Opening a remote hit (Open, Quick Look, drag-out) FETCHES the file from B on demand, behind a
  per-link switch on B ("Let <A> open files"). Linking therefore grants read access to what B
  indexes - and only to that: B serves a path only if its index holds it, never the clipboard
  history, never an arbitrary path.
- Same network (Bonjour) plus link-by-address. No relay, no rendezvous server.
- Ranking: B's hits are mixed into A's list when both indexes are in the same vector space (B's
  stored space probe against A's, cosine >= 0.999, same dimension); otherwise they are a separate
  group per Mac.

## Pairing

A 6-digit code is 20 bits. Used as a password or a TLS pre-shared key (Apple's TicTacToe sample
does that) anyone who records the pairing tries all 10^6 codes afterwards in about a second.
CryptoKit has no public PAKE. So the code is a SHORT AUTHENTICATION STRING with a commitment, the
construction Bluetooth's numeric comparison and ZRTP use, over a TLS 1.3 channel:

1. A (client) and B (server) open TLS 1.3 with each side's self-signed identity certificate; during
   pairing any key is accepted.
2. A sends `commit = SHA256(nA)` and its name. B answers with its random `nB`.
3. A reveals `nA`; B checks it against the commitment.
4. Both compute `code = SHA256("omni-link-sas" | tls-exporter | keyA | keyB | nA | nB)` -> 6 digits.
   A shows the code; B asks its user to type it. A match proves both saw each other's real key.

A man in the middle runs two TLS sessions and needs both codes equal. On the leg where it is the
server it must pick its nonce before A's is revealed; on the leg where it is the client it commits
before B picks. Either way one code is uniformly random to it: one chance in a million per attempt,
and nothing to work on afterwards. B accepts a pairing request only while its own "Link a Mac"
window is open, so nobody on the network can pop a sheet on a Mac that is not pairing.

Typing the code on B is B's consent to be searched by A. A's sheet carries "Also let <B> search this
Mac", A's consent. Each Mac switches its own grants later; neither can switch the other's.

## Sessions

- Each Mac has a long-term P-256 identity (`link/identity` in Omni's support folder, 0600 - the
  index next to it holds the content anyway). Its ID is the SHA-256 of the public key.
- Every connection is TLS 1.3 (Network.framework) with both certificates checked against the key
  pinned at pairing: forward secrecy, and the record layer is Apple's code. The only cryptographic
  construction we wrote is the pairing step above.
- Bonjour `_omni-link._tcp`: TXT carries the ID only; the service name is the Mac's name only while
  its Link window is open, the opaque ID otherwise.

## Search

- A sends the query text and its filters; B embeds with its own model and searches its own index,
  the same code path as a served search (clipboard history excluded). File queries and Find Similar
  send a vector, and only to Macs in the same vector space.
- Remote hits are `omni-link://<peer-id>/<path>` keys on A, handled like `photos://` keys:
  materialized on demand (fetched into a cache, read-only), thumbnails rendered by B.
- A's search never waits: remote calls run on their own queue, never on A's store queue, GPU or
  main thread; local results render as before and remote ones join when they arrive; a Mac past
  its deadline is skipped for that query. On B, one remote search per peer runs at a time
  (latest wins).

## Phases

1. Identity, pairing, the encrypted link, remote search merged into results, sidebar rows with status.
2. A linked Mac's folders as search scopes, the thumbnail cache, opening remote hits.
3. Browsing a linked Mac's folders without a query, Find Similar on a remote hit, `omni search --macs`.

## Proof required

Protocol tests on loopback (pairing, wrong code, a key-swapping relay, an unpaired key, a flipped
byte, an offline or frozen peer, cancel); two isolated instances end to end; A's search latency with
B killed, frozen and unplugged against the feature off; B's own search latency while A queries it.

## Findings (2026-10-12)

- TLS 1.3 with in-memory identities works in Network.framework: `SecIdentityCreate` is public on
  macOS, the certificate is a hand-built DER (v3, no extensions), and both verify blocks decide by
  the peer's key alone. In TLS 1.3 the CLIENT reports ready before the server has judged its
  certificate; a refused client learns it on its first request.
- `connect()` used to cancel every connection when its timer fired, connected or not: every link
  died 10 s in. Found in the two-instance run, not by the unit tests, whose calls all finished
  inside 10 s. `testLinkOutlivesConnectTimeout` now holds a link past the timer.
- A Mac that unlinked another while it was away could never link it again: the other still trusted
  it, so its pairing request reached the linked handler. Pairing messages now reach the pairing
  handler on every connection (`testRelinkWhenTheOtherMacStillTrustsThisOne`).
- The space probe is the loaded model's embedding of the probe text, not the index's stored probe:
  a new index has none until its first pass, which made two same-model Macs "different".
- Two isolated instances on one Mac (`-omni.link.port`, `-omni.link.name`, `-omni.link.bonjour`)
  pair through the real UI paths (perf script `linkpair` / `linkcode` / `linkaccept`). Same model:
  hits mix by score. Nano against small: the other Mac's hits are their own group under its name.
- Search to local results on A (40 distinct queries per arm, ms): link off p50 35.4/36.0, B online
  33.3, B killed 26.8, B frozen (SIGSTOP) 42.5. B's replies join at p50 ~80 ms. A frozen Mac is
  noticed by the ping (15 s + 5 s); a killed one at once.
- Footprint under a watchdog: A 3.2-3.4 GB, B (nano) 3.2-3.3 GB, B (small) 5.9-6.7 GB, flat across
  online and frozen arms and after a frozen B resumes with a backlog.
- `linktool` (a linked Mac with no app) against the real app: search, sources, listing, thumbnails
  and fetch answer (fetched bytes match by SHA-256); refused are /etc/passwd, ~/.ssh, a file on disk
  outside the index, a `..` path, a relative path, a listing of / or above a source or outside one,
  an unknown method, and every request touching the clipboard history (fetch, thumbnail, passages,
  Find Similar, listing, search hits, a search scoped to it). Positive control: a build without the
  clipboard guards returned the clipboard file in search and served its bytes. With "open files"
  off, search answers and fetch is refused.

## Agents (MCP and `omni`), 2026-10-12

- `search` covers linked Macs by default (Han); `macs` names some, `[]` or `none` keeps it to this
  Mac. A linked Mac's results come after this Mac's, under its name, with paths on that Mac; an
  offline one gets one line. A folder scope is a folder on this Mac, so it goes nowhere else.
- A served search gives linked Macs 3 s (the window gives them 10 s and never waits): with a Mac
  frozen, `omni search` returned in 3.3 s with "On Studio: No answer."
- `get_linked_file <path> --mac <name>` copies the file here (the same fetch and grant as Open) and
  prints the local, read-only path; path first because the command line takes the first required
  parameter bare. REST /v1/search is unchanged: local only.
- THE WAY TO THE FILE GOES WITH THE PATHS (2026-10-13). An agent searching over MCP got a linked
  Mac's results and reported it "can only see the search snippets": the only mention of
  `get_linked_file` was in the server instructions (many clients never show those to the model) and
  in the `macs` parameter's description. Each "On <Mac>:" header now says how to read its paths -
  `get_linked_file, mac "<Mac>"` over MCP, the full `omni get_linked_file <path> --mac "<Mac>"` on
  the command line - or that the Mac does not allow opening its files; the JSON carries `canOpen`.
  Checked on two linked instances, both grants, MCP over HTTP and the CLI.
- The sidebar folds each Mac like a History day (chevron on hover, kept while folded, state in
  SidebarFolds under "mac:<id>"); no disclosure triangle, which would indent every row.
