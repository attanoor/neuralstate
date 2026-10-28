"""How a real LLM agent uses NeuralState: recall before acting, record after.

    pip install "neuralstate[anthropic]"
    export ANTHROPIC_API_KEY=...
    python examples/claude_agent.py "Draft the migration plan for acme's database upgrade"

The memory block (learned rules first, then facts, then related episodes) goes into the
system prompt. The ids of the rules that were in context are sent back with the episode,
so NeuralState can measure whether each rule actually helps.
"""

import argparse
import os

import anthropic

from neuralstate.sdk import NeuralStateClient

SYSTEM = """You are an operations agent. Below is your long-term memory, maintained by a
background process that learns from your past work. Learned rules come from repeated
past failures: follow them unless the situation clearly differs, and say so if it does.

<memory>
{memory}
</memory>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--task-type", default="ops")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--agent", default="claude-ops-agent")
    args = parser.parse_args()

    ns = NeuralStateClient(args.url, agent=args.agent, api_key=os.environ.get("NS_API_KEY"))
    memory = ns.recall(args.task, task_type=args.task_type, budget_tokens=2000)

    client = anthropic.Anthropic()
    response = client.beta.messages.create(
        model="claude-opus-5-5",
        max_tokens=16000,
        system=SYSTEM.format(memory=memory["context"] or "(empty)"),
        messages=[{"role": "user", "content": args.task}],
        output_config={"effort": "medium"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        raise SystemExit("the model declined this task")
    answer = "".join(b.text for b in response.content if b.type == "text")
    print(answer)

    # In production the outcome comes from the environment (tests passed, the deploy succeeded,
    # the customer replied...). Recording "unknown" now and attaching the outcome later via
    # ns.set_outcome(episode_id, ...) is the normal pattern.
    episode = ns.record(
        summary=args.task, task_type=args.task_type, outcome="unknown",
        applied_rule_ids=memory["applied_rule_ids"],
        content={"messages": [{"role": "user", "content": args.task}, {"role": "assistant", "content": answer}]},
    )
    print(f"\n[recorded episode {episode['id']}; attach its outcome with set_outcome() when known]")


if __name__ == "__main__":
    main()
