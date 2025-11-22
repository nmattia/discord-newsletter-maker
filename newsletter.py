#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "openai",
# ]
# ///
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List
from typing import Sequence

from openai import APIConnectionError
from openai import APIError
from openai import APITimeoutError
from openai import AuthenticationError
from openai import OpenAI

SYSTEM_PROMPT = """
You are a newsletter editor for the newsletter of a maker community called 'The
Makery'. Read chat excerpts that contain shared links and their descriptions.

- Decide which links are worth including (educational, insightful, noteworthy).
- Drop broken or spammy links.
- Group related links together and keep things concise. Feel free to put the
  links in whatever order makes the most sense.
- Return an HTML list (ul.link-list > li) with a short title and bullets. Do mention a few words
  about each link, anything you can gather from its description and the messages in the
  context. Don't say things you aren't sure about, but do try to make it a bit less dry
  than just a link description.
- Include credit for who shared the link using the provided username (e.g., "Posted by username").
- Do not include a header, footer, or anything else apart from the list of links.
- Give the links the following structure:
  <li>
    <strong>Title with proper case</strong>
    <p>Description sentences/paragraphs.</p>
    <a href="https://link/to/the/page">https://link/to/the/page</a> <span class="poster">by Stavros.</span>
  </li>

""".strip()


def load_contexts(path: Path) -> List[dict]:
    """Load gathered link contexts from a JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    contexts = data.get("contexts") if isinstance(data, dict) else None
    if contexts is None and isinstance(data, list):
        contexts = data
    if not isinstance(contexts, list):
        raise SystemExit("Input JSON must include a 'contexts' array.")
    return contexts


def render_contexts(contexts: Sequence[dict]) -> str:
    """Turn structured contexts into a text prompt for the model."""
    lines: List[str] = []
    for context in contexts:
        source = context.get("source") or "unknown file"
        timestamp = context.get("timestamp") or "unknown time"
        lines.append(f"=== {source} @ {timestamp} ===")

        for message in context.get("messages") or []:
            author = message.get("author") or "Unknown"
            content = message.get("content") or ""
            message_lines = content.splitlines() or [""]
            lines.append(f"{author}: {message_lines[0]}")
            for line in message_lines[1:]:
                lines.append(f"    {line}")

        for link in context.get("links") or []:
            url = link.get("url") or ""
            posted_by = link.get("posted_by") or "Unknown"
            if url:
                lines.append(f"    [link] {url} (posted by {posted_by})")
            description = link.get("description") or ""
            if description:
                lines.append(f"    [description] {description}")

        lines.append("")

    return "\n".join(lines).strip()


def run_completion(client: OpenAI, model: str, context: str, temperature: float) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Create the newsletter from these Discord snippets:\n\n" + context
                ),
            },
        ],
    )
    choice = response.choices[0]
    return choice.message.content or ""


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Discord link dumps into a short newsletter."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the JSON file containing the gathered messages with links.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.1",
        help="OpenAI chat model to use (default: gpt-5.1).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Sampling temperature for the model (default: 0.4).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key (defaults to OPENAI_API_KEY env var).",
    )
    args = parser.parse_args(argv)

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing OpenAI API key. Set OPENAI_API_KEY or pass --api-key."
        )

    contexts = load_contexts(args.input)
    if not contexts:
        raise SystemExit("No link contexts found in input JSON.")
    context = render_contexts(contexts)

    client = OpenAI(api_key=api_key)

    try:
        newsletter = run_completion(
            client=client,
            model=args.model,
            context=context,
            temperature=args.temperature,
        ).strip()
    except (APIError, APIConnectionError, APITimeoutError, AuthenticationError) as exc:
        raise SystemExit(f"OpenAI API error: {exc}") from exc

    if not newsletter:
        raise SystemExit("Model returned an empty newsletter.")

    output = json.dumps({"LINK_CONTENT": newsletter})
    Path("newsletter_context.json").write_text(output, encoding="utf-8")
    print(newsletter)


if __name__ == "__main__":
    main()
