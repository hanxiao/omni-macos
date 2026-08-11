# Schema v4

What the index stores, and why it is shaped this way. Written against measurements from a real
2,361,404-chunk / 746,324-file index at dim 768, taken with `dbstat` before any of this existed.

## What v3 actually costs

    chunks                       615 MB   2.36M rows
      snippet                      425 MB
      chunk_key (32-char hex)       54 MB
      locator                       13 MB
      vec (99.9% empty)              1 MB
      numeric columns + headers    ~120 MB   <- per-FILE facts, stored per CHUNK
    content_keys + its 2 indexes 388 MB   746k rows
    files + autoindex            236 MB   746k paths, stored twice
    idx_media_snippet             38 MB
    chunks autoindex              31 MB
                                -------
                                1308 MB

Two facts dominate and neither is the vectors:

  PATHS ARE STORED FOUR TIMES. Average path is 151 bytes. `files` holds one copy, its UNIQUE
  autoindex a second, `content_keys.path` a third, and that table's autoindex a fourth: ~460 MB,
  35% of the database, to say 746k things once.

  PER-FILE FACTS ARE STORED PER CHUNK. `modified`, `size`, `kind`, `width`, `height`, `duration`,
  `indexed_at` describe the FILE. At 3.16 chunks per file they are written 3.16 times each.

And one structural fault: the `vec` column. A vector is written into the row, later cleared once
the `.vecs` file covers it, and the freed bytes stay inside a page that stays allocated. The
database hollows out for as long as the app indexes - which is what the launch-time repack exists
to undo. That is a loop, not a fix: overclaim, reclaim, repeat.

## v4

    dirs(id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE)

    files(id INTEGER PRIMARY KEY, dir_id INTEGER NOT NULL, name TEXT NOT NULL,
          modified REAL NOT NULL, size INTEGER NOT NULL DEFAULT 0,
          kind INTEGER NOT NULL DEFAULT 0,
          width INTEGER NOT NULL DEFAULT 0, height INTEGER NOT NULL DEFAULT 0,
          duration REAL NOT NULL DEFAULT 0, indexed_at REAL NOT NULL DEFAULT 0)
    UNIQUE INDEX idx_files_name ON files(dir_id, name)

    chunks(id INTEGER PRIMARY KEY, file_id INTEGER NOT NULL,
           chunk_index INTEGER NOT NULL, kind INTEGER NOT NULL)
    UNIQUE INDEX idx_chunk_file ON chunks(file_id, chunk_index)

    chunk_text(chunk_id INTEGER PRIMARY KEY, kind INTEGER NOT NULL, file_id INTEGER NOT NULL,
               snippet TEXT NOT NULL, locator TEXT NOT NULL DEFAULT '',
               chunk_key BLOB NOT NULL DEFAULT x'')
    INDEX idx_chunk_label ON chunk_text(kind, snippet, file_id) WHERE kind IN (media)

    pending_vecs(chunk_id INTEGER PRIMARY KEY, vec BLOB NOT NULL)

    dedup(file_id INTEGER PRIMARY KEY, key BLOB NOT NULL,
          modified REAL NOT NULL, size INTEGER NOT NULL)
    INDEX idx_dedup_key ON dedup(key)

    kinds(code INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE)

### What it measured, on the index the numbers above came from

The conversion was replayed on a copy of a live 2,361,404-chunk / 746,324-file index at dim 768.

    database                    1311 MB  ->   786 MB     -40%
      chunks + its index          646    ->    64
      chunk_text + label index      -    ->   539         (425 MB of it is snippets)
      files + index               236    ->    77
      dirs + index                  -    ->    58
      content_keys / dedup        388    ->    41

    what a cold load must scan    733 MB  ->   109 MB     -85%
    conversion, one time                     26.1 s

    open, warm page cache        2.43 s  ->  2.42 s
    open, row sidecar adopted    0.70 s  ->  0.70 s
    search p50 / p95 (topK 40)   2.9 / 211 ms -> 2.9 / 199 ms
    folder delete (6,553 files)   151 ms  ->   147 ms
    batched write            7641 files/s -> 6494 files/s   -15%

The write is the one that got slower, and structurally so: a chunk is three rows now (the row, its
text, its pending vector) where it was one. It costs 0.023 ms per file against tens of ms of
embedding per file, so it is ~0.1% of an indexing pass - paid for by the 40% and by the load scan,
which is the number that decides how long a launch takes on a machine whose page cache does not
already hold the index.

### chunks is now narrow, and its id is stable

`id INTEGER PRIMARY KEY` is not decoration. In v3 `chunks` had no explicit rowid alias, so VACUUM
was free to renumber - the whole safety argument for the launch repack was that it preserves row
ORDER even though it changes rowid VALUES. Every side table here keys on `chunk_id`, and that
argument does not stretch to cover a join. An explicit INTEGER PRIMARY KEY is preserved by VACUUM
outright, so the side tables cannot be silently misaligned by maintenance.

The slot rule is unchanged: a row's slot in `.vecs` is its rank in id order, counted through the
holes in `vec_holes`.

### pending_vecs, and why the hollowing stops

A vector is durable in SQLite until `.vecs` is msync'd and coverage reaches it. That has to stay
true - it is the only thing making a crash mid-index survivable. What changes is WHERE the durable
copy lives. In a dedicated table, clearing coverage is `DELETE FROM pending_vecs WHERE chunk_id <=
B`, which frees whole pages onto the freelist, and the next batch of pending vectors takes those
same pages back. Steady state, the file does not grow from this at all, and there is nothing for a
launch-time repack to reclaim.

It also makes the advance O(slice) instead of O(covered). v3 re-walked the whole covered prefix on
every slice (`UPDATE chunks SET vec = x'' ... ORDER BY rowid LIMIT clearUpTo`) because a rowid
watermark had been tried, got the slot arithmetic wrong, and was reverted. With a stable id the
watermark is not arithmetic on slot counts: `vecs_covered_id` is committed in the same transaction
as `vecs_covered_rows`, and it is derivable from scratch if lost.

### Dirs are interned, paths are not stored again

220,510 distinct directories against 746,324 files: 27 MB of directory text and 21 MB of basenames,
versus 113 MB of full paths. `dedup` drops its path column entirely and keys on `file_id`, which
makes it a rowid table with no autoindex at all.

Prefix queries get better, not worse. `deleteUnderFolder` was a range scan over 113 MB of path
text; it is now a range scan over 27 MB of directory text, followed by an id join.

### Content keys are 16 bytes, not 95 characters

The dedup key is a composite string - `2|audio|flac|m768|s33280281|s24000|<64 hex>` - averaging 95
bytes, indexed, at 746k rows. Stored as the first 16 bytes of its SHA-256 it is a BLOB: 128 bits
over 746k items is a collision every ~10^24 indexes, and a false positive is caught anyway by the
existing lockstep check against `modified`.

## Migration

Phased and resumable, with the old tables untouched until one small final transaction. Each phase
commits in batches, so a kill at any point resumes from the last committed batch rather than
restarting - and there is no state in which the index is half of each schema.

    0  create the v4 tables empty; clear any partial ones from an abandoned attempt
    1  dirs, then files (paths split; per-file facts folded in from chunks)
    2  chunks + chunk_text, in id order, batched          <- resumable at max(id) already copied
    3  pending_vecs, from the rows that still carry a blob
    4  dedup, from content_keys
    5  verify, then ONE transaction: drop v3 tables, rename, user_version = 4

Phase 2 preserves id order exactly: new ids are the old rowids. That is what keeps every coverage
claim, every hole and every slot in `.vecs` valid across the migration, with nothing re-embedded.

A 0.4.x index reaches v4 the same way it reaches v3 today - the legacy path-interning conversion
runs first, then this. Neither re-embeds.

Disk: both copies exist at once, so the migration needs roughly the database's size free and
declines (with a message, not an error) when it is not there.

### Downgrade

An older binary opening a v4 index sees `user_version = 4` and drops `chunks` and `content_keys` -
but NOT `files`, whose name it shares with a table of an incompatible shape. Its
`CREATE TABLE IF NOT EXISTS files(id, path)` then no-ops, and it can neither read `f.path` nor
insert a row (`dir_id` is NOT NULL), so it indexes nothing at all. It does not merely reindex.

Coming back is what has to work, and it is handled explicitly: "chunks is v2/v3 AND files has
`dir_id`" can only be a downgrade round trip, so that `files` is dropped before the conversion runs.
Without it, phase 1 selects `f.path` from a v4-shaped table and fails on every launch, with Repair
unable to help - it only knows how to drop a path table under a LEGACY chunk table.

The lesson is in the table names: reusing `files` and `chunks` for incompatible shapes is what makes
a downgrade a one-way door rather than a reindex.
