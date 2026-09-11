# Agent command catalog

| Command | Purpose | Delegates to |
| --- | --- | --- |
| `#dwp-create <goal>` | Create a Lite-first DWP plan; Full only when needed | `deepworkplan-create` |
| `#dwp-execute` | Execute the next eligible plan task with validation | `deepworkplan-execute` |
| `#dwp-status` | Report plan state without modifying work | `deepworkplan-status` |
| `#dwp-refine` | Safely change scope or promote Lite → Full | `deepworkplan-refine` |
| `#dwp-resume` | Resume an interrupted plan from durable state | `deepworkplan-resume` |
| `#dwp-verify` | Verify DWP conformance and plan structure | `deepworkplan-verify` |

The command files in `.agents/commands/` are intentionally thin. DWP lifecycle behavior is defined by the installed skill, not copied into this repository.
