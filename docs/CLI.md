# The omni command line

`omni` lets agents and scripts use the running Omni app from a shell. It never loads a model: it
talks to the app, starting it if it is not running.

## One interface, three ways in

The MCP tools in `App/Serving/MCPAdapter.swift` are the interface. Everything else reads them:

- MCP clients call them over `POST /mcp`.
- `omni` is a generic MCP client. It asks for `tools/list` and turns each tool into a subcommand
  whose flags are the tool's parameters (`top_k` is `--top-k`, a list repeats its flag, a boolean
  is `--flag` / `--no-flag`, and the first required parameter takes the bare words). A tool or
  parameter added to the app shows up here with no CLI change.
- SKILL.md is rendered from the server's `initialize` instructions plus `tools/list`
  (`Sources/OmniCLI/AgentSkill.swift`, compiled into both the app and the CLI). `omni skill` and
  Settings > Serving > SKILL.md print the same bytes.
- REST `/v1/search` reads its `filters` with the same code as the MCP tool
  (`App/Serving/SearchArgs.swift`); only the output format differs.

## History

A search from the command line is recorded with source `cli` and drawn with a terminal icon in the
History drawer, beside REST (globe) and MCP clients (the MCP mark). It is an MCP call, so the
surface is decided by transport: the socket's router reports `.cli`.

## Transport

The app serves its HTTP router on a Unix socket at
`~/Library/Application Support/Omni/omni.sock` whenever a model is loaded. This is independent of
the Serving toggle, which opens a TCP port. The socket is mode 0600, so only the user can connect,
and it needs no token. A socket left by a crash is cleared on the next launch. A live one belonging
to another Omni is left alone. Isolated runs (`-omni.dbDir`) get no socket unless they pass
`-omni.socketPath`.

`omni --url http://host:port --token T` uses HTTP instead, for another Mac's Serving on the LAN.

## Starting the app

When there is no socket and no Omni process, `omni` launches the app it ships in, in the
background (`open -g -j`), and waits up to 3 minutes. If Omni is running but has no socket (still
loading, or a version from before the socket), it does NOT run `open`: that would send the app a
reopen event, which shows its window. It waits 60 s, then says to update Omni.

## Shipping

Built from `Sources/OmniCLI` by the `omni-cli` tool target in `project.yml`, signed with the app,
and embedded at `Omni.app/Contents/Helpers/omni`, not `Contents/MacOS`, because `omni` and `Omni`
are the same file on a case-insensitive volume. Settings > Serving > Install links it into
`/usr/local/bin`, asking for an administrator password when that folder is not writable. The
skill uses the absolute path, so agents need no install.

## Design principles, and how each is met

From "CLI is All Agents Need", reviewed 2026-10-07:

- One tool. `omni` is one command; the skill names its subcommands and shows examples, and every
  option stays behind `--help`. The skill went from 125 lines (every flag of every tool) to 44.
- stdout is data, stderr diagnostics. The server marks blocks that are about the answer rather
  than the answer - "No results for ...", the empty-index note, the image cap, the OCR page range
  - with `_meta` `io.hanxiao.omni/stderr`, only on calls the CLI marks with
  `io.hanxiao.omni/cli`. So `omni ocr scan.pdf > scan.md` is Markdown, and an empty search leaves
  stdout empty and explains itself on stderr. Other MCP clients get the same bytes as before.
- Progressive help. `omni --help`: the commands and two examples. `omni <tool> --help`: usage, the
  first sentence, examples, one line per option. `--help-all`: every description in full.
  Examples live once, as argument sets in each tool's `_meta`, and are rendered as commands. A
  boolean that defaults on (JSON Schema `default`) shows as `--[no-]flag`.
- Errors that course-correct. Unknown commands, options and enum values suggest the closest
  match; a missing argument shows usage and an example; server errors are reworded into the
  CLI's terms (`'modified_after'` becomes `--modified-after`, `'query'` becomes `<query>`).
- Consistent output. Plain text is the same "N. path  (meta)" lines MCP clients read; `--json` is
  structuredContent, built by the same row builders as REST, so `omni search --json` and
  `/v1/search` have the same fields.
- Exit codes: 0 done, 1 the command needs fixing (bad option, or the tool rejected an input),
  2 Omni unavailable or failed, 130 interrupted (default SIGINT, measured as a kill by signal 2).
- Raw at the pipe: blocks are joined by single newlines; `--json` carries full snippets, the text
  output a preview set by `--max-snippet`.

Not met: plain output is meant for reading, so a script wanting paths uses `--json` (`omni search
x --json | jq -r '.results[].path'`) rather than cutting lines.

## Commands

    omni search "invoice from february" --top-k 5 --modified-after 2026-02-01 --ext pdf
    omni file_status /path/a.pdf /path/b.png
    omni tools            # the tool list
    omni <tool> --help    # a tool's flags, from its schema
    omni skill            # SKILL.md
    --json                # the raw MCP result instead of text

Exit codes: 0 done, 1 fix the command, 2 Omni unavailable, 130 interrupted.
