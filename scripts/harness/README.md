# OpenCode harness

The acceptance test. Everything else in `scripts/` checks that a specific mechanism
works — this checks that a **real agentic client** can hold a multi-turn conversation
with tool calls against the server, which is how the server is actually used.

```bash
./scripts/harness/run.sh                                  # all prompts, server on :8081
./scripts/harness/run.sh --url http://127.0.0.1:8099      # a different server
./scripts/harness/run.sh --model ref/occamy               # the llama.cpp baseline provider
./scripts/harness/run.sh --only 2                         # one prompt
```

Results land in `runs/<UTC timestamp>/` — the per-prompt `opencode --format json`
stream, stderr, and a `summary.md` table. Never `/tmp`: it is tmpfs on this machine,
so anything written there lives in RAM instead of on the NVMe.

## How it works

`run.sh` does **not** start a server. It health-checks one and refuses to run if none
answers, and it refuses if `amp-server` *and* `llama-server` are both up — they contend
for the same 6 GB of VRAM and every number becomes meaningless. A stale
`amp-server` holds ~5.4 GB, so check `pgrep -a amp-server` before blaming a result.

Each prompt runs in a **pristine copy** of `workspace/`, because prompts 2 and 3 edit
files. Without that, a second run of the suite would start from an already-fixed tree
and the results would not be comparable to the first run.

The provider config is generated into the run directory as `opencode.json`, next to the
workspace rather than inside it. OpenCode discovers config by walking up from the working
directory, so this keeps the fixture clean. Its shape (`package` + `settings.baseURL`)
is copied from the user's known-working `~/.config/opencode/opencode.json`; do not
"modernise" it to `npm`/`options` without checking the providers guide.

`--auto` is mandatory. Without it OpenCode stops on the first tool call waiting for a
permission answer a script cannot give, and the run hangs forever.

## The prompts

Each one targets a failure mode that actually reached real use, not just "does a tool
call work".

| prompt | what it exercises |
|---|---|
| `01-read-and-report` | the baseline: one tool call, a tool result round trip, and a read-only answer. Fails if basic tool-call emission or parsing is broken. |
| `02-fix-failing-test` | **the agentic pattern, and the prefix-cache regression.** The model must run the tests, read the failure, diagnose it, edit the source, and re-run. This is the multi-turn case where a thinking model that forgets to send its reasoning back re-evaluates the whole conversation every turn — the single most expensive bug in this project's history. |
| `03-add-feature-and-test` | sustained multi-step work: a new function, a new test, a full run, and recovery if a test it wrote fails. Longer context growth and more tool round trips than 02. |

`workspace/` is a tiny Python project with one deliberate bug, so finding it requires
actually running the tests. It is plain-stdlib and needs no `pytest`, so the model never
has to install anything mid-run.
