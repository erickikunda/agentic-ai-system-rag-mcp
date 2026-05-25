# Agentic Workflow Notes

Agentic systems become easier to operate when planning, retrieval, tool use,
verification, and escalation are explicit states instead of branches hidden
inside a single prompt. The graph does not make the model smarter; it makes the
system easier to inspect, interrupt, and recover.

Human review is most useful at boundary decisions: accepting a plan, approving
an irreversible tool call, or deciding whether weak evidence is enough for a
high-stakes answer. Interrupts should preserve state so the workflow can resume
without replaying expensive model calls.

Tool descriptions are part of the control plane. A vague tool schema creates
avoidable model confusion, while a narrow schema with clear input constraints
reduces invalid calls and simplifies retry handling.

