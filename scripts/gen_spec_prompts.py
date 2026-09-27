#!/usr/bin/env python3
"""Generate the two speculative-decoding workloads from live repository source.

    python3 scripts/gen_spec_prompts.py

Separate from scripts/bench_spec.py on purpose. The workload is a *fixture*, and a fixture that
lives inside the benchmark is a fixture that silently goes stale: the last version of the code
prompt embedded a copy of model_configs_path() that had already been rewritten, so it was
asking the model to reproduce a function that no longer exists.

The code workload's whole point is that the answer must *repeat text already in the context*,
because that is the only condition under which n-gram speculation can win. A prompt about
something abstract measures nothing. Keeping it derived from a real file also keeps it
representative: this is what a coding agent's turn actually looks like.

Two workloads, and the second is not optional. Strata measured prompt-lookup speculation at
6-11% on code edits and unchanged on prose, and separately measured that *forcing* the lookup
drafter whenever it proposed more than the draft model lost 2-8% on ordinary text. So a
single-workload benchmark would report a win that is a loss on the other half of the work.
"""
import pathlib
import sys

# Read from the repository, not a copy. The prompt quotes this file.
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent

CODE_INTRO = """Here is a function from a C++ file I am working on.

```cpp
{fn}
```

I need to change how it finds the config file. Two problems with the current version:

1. When the executable is not under `build/bin/`, the `find_last_of('/')` calls do not strip
   the right number of path components, so it returns a path that does not exist and the
   caller silently does nothing.
2. There is no way to point it somewhere else for testing without editing and recompiling.

Please rewrite the function so that it:

- takes the repository root as an argument instead of deriving it, with a default that still
  does the `/proc/self/exe` walk when no argument is given,
- honours an environment variable override if it is set,
- returns `std::optional<std::string>` rather than a bare path, and
- keeps the existing `/proc/self/exe` behaviour for the no-argument case.

Reproduce the whole function in your answer, with a short comment above each non-obvious
branch, exactly as it would appear in the file."""

PROSE = """I am trying to understand how memory bandwidth limits a mixture-of-experts model on a
consumer graphics card, and I keep going back and forth with myself about the arithmetic.

Please explain, in about four paragraphs of plain prose with no code and no bullet points,
why moving the "hot" expert matrices into VRAM does not automatically make decoding faster
even though it removes most of the bytes the CPU has to read. I understand that reading
memory is the bottleneck. What I do not understand is why removing two thirds of the bytes
does not remove two thirds of the time.

Please reason it through carefully and at length, the way you would explain it to a
colleague who is smart but has never worked on an inference engine. Take your time, and
explain the reasoning as you go rather than just stating conclusions."""


def extract_function(text, signature):
    """The text of one top-level function, by brace matching from its signature.

    Counting braces rather than looking for a fixed offset or an end marker, so a later edit
    to the file cannot leave this quietly quoting the wrong span.
    """
    i = text.index(signature)
    j = text.index("{", i)
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[i:k + 1]
    raise ValueError(f"unbalanced braces after {signature!r} at offset {i}")


def main():
    out_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode")
    out_dir.mkdir(parents=True, exist_ok=True)

    src = (REPO / "src" / "plan" / "preflight.cpp").read_text()
    fn = extract_function(src, "static std::string model_configs_path()")

    code = CODE_INTRO.format(fn=fn)
    (out_dir / "spec_code.txt").write_text(code)
    (out_dir / "spec_prose.txt").write_text(PROSE)

    print(f"code  {len(code):6d} chars  (embeds a {len(fn)}-char function, "
          f"{len(fn.splitlines())} lines)")
    print(f"prose {len(PROSE):6d} chars")
    print(f"written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
