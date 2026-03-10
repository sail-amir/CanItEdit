#!/usr/bin/env -S uv run --script
# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "aiohttp",
#     "tqdm",
#     "datasets==4.0.*",
# ]
# ///
"""
Custom script for CanItEdit benchmark evaluation with custom endpoint support.
Supports custom headers (csb-token) and both streaming/non-streaming modes.
"""

import datasets
from pathlib import Path
from tqdm import tqdm
from typing import List, Literal, Optional, TypedDict
import gzip
import json
import itertools
import asyncio
import aiohttp


def gunzip_json_write(path: Path, data: dict) -> None:
    with gzip.open(path, "wt") as f:
        json.dump(data, f)


Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


# this is one edit request
class EditCommand(TypedDict):
    instruction: Optional[str]
    content: str


# this is model's output
class EditResponse(TypedDict):
    instruction: Optional[str]
    content: str


def chat_edit_prompt_zeroshot(old: str, instr: str) -> List[Message]:
    return [
        {
            "role": "system",
            "content": """
You are PythonEditGPT. You will be provided the original code snippet and an instruction that specifies the changes you need to make. You will produce the changed code, based on the original code and the instruction given. Only produce the code, do not include any additional prose.
            """.strip(),
        },
        {
            "role": "user",
            "content": f"""
## Code Before
```py
{old}
```

## Instruction
{instr}

## Code After""".strip(),
        },
    ]


# Copied from prl_ml
def extract_code_from_markdown(markdown):
    """
    Extracts the first markdown block of code from markdown.

    Strips away the language tag on the first line if present. Supports markdown
    that has several code blocks (just returns the first).
    """
    # Find the first code block
    code_block_start = markdown.find("```")
    if code_block_start == -1:
        return None

    # Skip past the opening ```
    code_start = code_block_start + 3

    # Find the end of this code block
    code_block_end = markdown.find("```", code_start)
    if code_block_end == -1:
        return None

    # Extract the code between the markers
    code = markdown[code_start:code_block_end].strip()

    if "# Example usage:" in code:
        code = code.split("# Example usage:")[0]

    # Remove language tag if present on first line
    first_newline = code.find('\n')
    if first_newline > 0:
        code = code[first_newline + 1:]

    return code.strip()


# Note that the code below is not fenced.
ONE_SHOT_EXAMPLE = {
    "before": """def add(a, b):
    return a + b""",
    "instruction": """Add a "sub" function that subtracts two numbers. Also write docstrings for both functions and change a,b to x,y.""",
    "after": '''def add(x, y):
    """Adds two numbers."""
    return x + y

def sub(x, y):
    """Subtracts two numbers."""
    return x - y'''
}


class CustomEndpointChatModel:
    """
    Custom model class for endpoints with custom headers and streaming support.
    Designed for endpoints like:
    http://onlineservice.cn-southwest-2:8087/csb-inner-service/rest/infers/.../v1/chat/completions
    """

    def __init__(
        self,
        api_url: str,
        model_name: str,
        csb_token: str,
        auth_token: str = "nokey",
        use_streaming: bool = False,
        one_shot: bool = False,
    ):
        self.api_url = api_url
        self.model_name = model_name
        self.csb_token = csb_token
        self.auth_token = auth_token
        self.use_streaming = use_streaming
        self.one_shot = one_shot

    def _format_messages(self, old: str, instr: str) -> List[Message]:
        """Format the edit prompt as chat messages"""
        messages = chat_edit_prompt_zeroshot(old, instr)

        if self.one_shot:
            # Insert one-shot example before the actual task
            example_messages = [
                {
                    "role": "user",
                    "content": f"""## Code Before
```py
{ONE_SHOT_EXAMPLE['before']}
```

## Instruction
{ONE_SHOT_EXAMPLE['instruction']}

## Code After""",
                },
                {
                    "role": "assistant",
                    "content": f"```py\n{ONE_SHOT_EXAMPLE['after']}\n```",
                }
            ]
            # Insert after system message
            messages = [messages[0]] + example_messages + [messages[1]]

        return messages

    async def generate_non_streaming(
        self,
        prompt: EditCommand,
        session: aiohttp.ClientSession,
        **kwargs
    ) -> str:
        """Generate completion without streaming"""
        assert prompt["instruction"] is not None, "Instruction is required"

        messages = self._format_messages(prompt["content"], prompt["instruction"])

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.8),
            "top_k": kwargs.get("top_k", 20),
            "max_tokens": kwargs.get("max_tokens", 3072),
            "stream": False,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth_token}",
            "csb-token": self.csb_token,
        }

        async with session.post(self.api_url, json=payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"API Error {response.status}: {error_text}")

            result = await response.json()
            content = result["choices"][0]["message"]["content"]

            # Extract code from markdown if present
            code = extract_code_from_markdown(content)
            return code if code else content

    async def generate_streaming(
        self,
        prompt: EditCommand,
        session: aiohttp.ClientSession,
        **kwargs
    ) -> str:
        """Generate completion with streaming"""
        assert prompt["instruction"] is not None, "Instruction is required"

        messages = self._format_messages(prompt["content"], prompt["instruction"])

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.8),
            "top_k": kwargs.get("top_k", 20),
            "max_tokens": kwargs.get("max_tokens", 3072),
            "stream": True,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth_token}",
            "csb-token": self.csb_token,
        }

        full_content = ""

        async with session.post(self.api_url, json=payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"API Error {response.status}: {error_text}")

            # Read streaming response
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if not line or line == "data: [DONE]":
                    continue

                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])  # Remove "data: " prefix
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            content_chunk = delta.get("content", "")
                            full_content += content_chunk
                    except json.JSONDecodeError:
                        continue

        # Extract code from markdown if present
        code = extract_code_from_markdown(full_content)
        return code if code else full_content

    async def generate(self, prompt: EditCommand, **kwargs) -> str:
        """Generate completion (delegates to streaming or non-streaming)"""
        # Session is passed via kwargs
        session = kwargs.pop("_session")

        if self.use_streaming:
            return await self.generate_streaming(prompt, session, **kwargs)
        else:
            return await self.generate_non_streaming(prompt, session, **kwargs)


async def process_example_and_instruction(
    ex: dict,
    instr_kind: str,
    model: CustomEndpointChatModel,
    model_kwargs: dict,
    args,
    output_dir: Path,
    batch_sema: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    pbar: tqdm,
) -> None:
    """
    Process a single example and instruction kind by generating completions and saving results.
    """
    # Resume support
    path = output_dir / f"{ex['full_name']}_{instr_kind}.json.gz"
    if path.exists():
        pbar.update(1)
        return

    example = EditCommand(instruction=ex[instr_kind], content=ex["before"])

    async def gen(example):
        """
        Issues a request, but concurrency limited by batch_sema.
        """
        async with batch_sema:
            try:
                async with asyncio.timeout(args.timeout):
                    return await model.generate(example, _session=session, **model_kwargs)
            except asyncio.TimeoutError:
                return "<timeout encountered>"
            except Exception as e:
                return f"<error: {str(e)}>"

    completion_tasks = [gen(example) for _ in range(args.completion_limit)]
    completions = await asyncio.gather(*completion_tasks)

    # copy over the example
    result = {}
    for k in ex:
        result[k] = ex[k]

    result["instr_kind"] = instr_kind
    result["prompt"] = ""
    result["completions"] = completions
    result["language"] = "py"
    result["temperature"] = args.temperature
    result["top_p"] = args.top_p
    result["top_k"] = args.top_k
    result["max_tokens"] = args.max_tokens
    result["stream"] = args.stream
    result["script_args"] = args.__dict__.copy()

    gunzip_json_write(path, result)
    pbar.update(1)


async def main(args):
    dataset = datasets.load_dataset(args.dataset, args.subset, split=args.split)

    # Create custom model
    model = CustomEndpointChatModel(
        api_url=args.api_url,
        model_name=args.model,
        csb_token=args.csb_token,
        auth_token=args.auth_token,
        use_streaming=args.stream,
        one_shot=args.one_shot,
    )

    model_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
    }

    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True)

    instr_kinds = ["instruction_descriptive", "instruction_lazy"]
    items = list(itertools.product(dataset, instr_kinds))

    batch_sema = asyncio.Semaphore(args.batch_size)
    pbar = tqdm(total=len(items))

    # Create persistent session for connection pooling
    timeout = aiohttp.ClientTimeout(total=args.timeout + 10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with asyncio.TaskGroup() as tg:
            for ex, instr_kind in items:
                tg.create_task(
                    process_example_and_instruction(
                        ex,
                        instr_kind,
                        model,
                        model_kwargs,
                        args,
                        Path(args.output_dir),
                        batch_sema=batch_sema,
                        session=session,
                        pbar=pbar,
                    )
                )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CanItEdit benchmark with custom endpoint support"
    )
    parser.add_argument(
        "--dataset", type=str, default="nuprl/CanItEdit", help="dataset to use"
    )
    parser.add_argument(
        "--split", type=str, default="test", help="split of the dataset to use"
    )
    parser.add_argument(
        "--subset", type=str, default=None, help="subset of the split to use"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        required=True,
        help='Full API URL (e.g., "http://host:8087/csb-inner-service/rest/infers/.../v1/chat/completions")',
    )
    parser.add_argument(
        "--model",
        type=str,
        default="pangu_auto",
        help="model name to use in requests",
    )
    parser.add_argument(
        "--csb-token",
        type=str,
        required=True,
        help="CSB token for authentication",
    )
    parser.add_argument(
        "--auth-token",
        type=str,
        default="nokey",
        help='Bearer token for Authorization header (default: "nokey")',
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming mode (default: False)",
    )
    parser.add_argument(
        "--one-shot",
        action="store_true",
        help="Use one-shot prompting with example (default: False)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="output directory for completions"
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="concurrent requests limit"
    )
    parser.add_argument(
        "--completion-limit",
        type=int,
        default=20,
        help="number of completions per prompt",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="sampling temperature"
    )
    parser.add_argument("--top-p", type=float, default=0.8, help="top-p sampling")
    parser.add_argument("--top-k", type=int, default=20, help="top-k sampling")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3072,
        help="max new tokens to generate per completion",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="timeout in seconds for each request (default: 60)",
    )
    args = parser.parse_args()
    asyncio.run(main(args))
