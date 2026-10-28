"""Run the learning demo against a NeuralState API that is already running.

    neuralstate api            # in one terminal (and optionally `neuralstate worker`)
    python examples/demo_agent.py --url http://127.0.0.1:8000

`neuralstate demo` does the same and starts an API for you if none is running.
"""

import argparse
import os

from neuralstate import demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--agent", default=None)
    args = parser.parse_args()
    demo.run(args.url, api_key=os.environ.get("NS_API_KEY"), agent_name=args.agent)
