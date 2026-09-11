# Agent command catalog

| Command | Purpose | Route in installed DWP pack |
| --- | --- | --- |
| `#dwp-create <goal>` | Create a DWP plan; model development requests Full explicitly | `deepworkplan` → `create` |
| `#dwp-execute` | Execute the next eligible plan task with validation | `deepworkplan` → `execute` |
| `#dwp-status` | Report plan state without modifying work | `deepworkplan` → `status` |
| `#dwp-refine` | Safely change scope or promote Lite → Full | `deepworkplan` → `refine` |
| `#dwp-resume` | Resume an interrupted plan from durable state | `deepworkplan` → `resume` |
| `#dwp-verify` | Verify DWP conformance and plan structure | `deepworkplan` → `verify` |

The command files in `.agents/commands/` are intentionally thin. They route through the installed root `deepworkplan` skill so the repository does not depend on every sub-skill being separately linked by a host installer. DWP lifecycle behavior remains owned by the installed skill, not copied into this repository.
