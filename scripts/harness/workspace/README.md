# Fixture project for the OpenCode harness

A tiny Python project with one deliberate bug. It exists so the harness prompts have
something real to read, edit and verify, without depending on any large repository.

- `src/inventory.py` — four small functions, one of which is wrong on purpose
  (`total_value_cents` returns a float where the name and its callers imply exact
  integer cents).
- `tests/test_inventory.py` — plain-stdlib tests, runnable with
  `python3 tests/test_inventory.py`. No pytest dependency, so the model does not
  have to install anything to verify a fix.

Keep it small. The point is to exercise tool calls cheaply, not to be a real project.
