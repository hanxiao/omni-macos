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

## Commands

    omni search "invoice from february" --top-k 5 --modified-after 2026-02-01 --ext pdf
    omni file_status /path/a.pdf /path/b.png
    omni tools            # the tool list
    omni <tool> --help    # a tool's flags, from its schema
    omni skill            # SKILL.md
    --json                # the raw MCP result instead of text

Exit codes: 0 ok, 1 the tool reported an error, 64 bad arguments, 69 Omni unavailable, 77 refused
token.
