"""Thin wrapper over Bedrock Converse API with a tool-calling loop."""
import json
import logging
import re
from typing import Callable, Iterator

import boto3
from botocore.config import Config

from ..config import AWS_REGION, BEDROCK_MODEL_ID

log = logging.getLogger(__name__)
_client = boto3.client(
    "bedrock-runtime", region_name=AWS_REGION,
    config=Config(read_timeout=180, retries={"max_attempts": 4, "mode": "adaptive"}),
)


def tool_config(specs: list[dict]) -> dict:
    return {"tools": [{"toolSpec": {"name": s["name"], "description": s["description"],
                                    "inputSchema": {"json": s["schema"]}}} for s in specs]}


def run_agent(system: str, messages: list[dict], specs: list[dict], handlers: dict[str, Callable],
              max_steps: int = 8, max_tokens: int = 2500) -> Iterator[dict]:
    """Yields events: {"type": "tool", ...} for each tool call, then {"type": "final", "text": ...}."""
    cfg = tool_config(specs) if specs else None
    for _ in range(max_steps):
        kw = dict(modelId=BEDROCK_MODEL_ID, system=[{"text": system}], messages=messages,
                  inferenceConfig={"maxTokens": max_tokens, "temperature": 0.2})
        if cfg:
            kw["toolConfig"] = cfg
        resp = _client.converse(**kw)
        msg = resp["output"]["message"]
        messages.append(msg)
        calls = [c["toolUse"] for c in msg["content"] if "toolUse" in c]
        if not calls:
            text = "".join(c.get("text", "") for c in msg["content"]).strip()
            yield {"type": "final", "text": text}
            return
        results = []
        for call in calls:
            name, args = call["name"], call.get("input") or {}
            yield {"type": "tool", "name": name, "input": args}
            try:
                out = handlers[name](**args)
                status = "success"
            except Exception as e:  # report tool errors back to the model
                out, status = {"error": str(e)}, "error"
            results.append({"toolResult": {"toolUseId": call["toolUseId"], "status": status,
                                           "content": [{"text": json.dumps(out, default=str)[:12000]}]}})
        messages.append({"role": "user", "content": results})
    # Out of steps: ask for the answer without tools
    messages.append({"role": "user", "content": [{"text": "Stop using tools. Give your final answer now."}]})
    resp = _client.converse(modelId=BEDROCK_MODEL_ID, system=[{"text": system}], messages=messages,
                            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.2},
                            **({"toolConfig": cfg} if cfg else {}))
    text = "".join(c.get("text", "") for c in resp["output"]["message"]["content"]).strip()
    yield {"type": "final", "text": text}


def extract_json(text: str) -> dict:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S) or re.search(r"(\{.*\})", text, re.S)
    if not m:
        raise ValueError(f"No JSON in model output: {text[:200]}")
    return json.loads(m.group(1))
