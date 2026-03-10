# Custom Endpoint Usage Guide

This guide explains how to use `custom_generate_completions.py` with your custom API endpoint.

## Features

✅ **Custom Headers**: Supports `csb-token` and custom `Authorization` headers
✅ **Streaming Support**: Toggle between streaming (`stream=true`) and non-streaming (`stream=false`)
✅ **One-Shot Prompting**: Optional one-shot example for better performance
✅ **Async/Batch Processing**: Efficient concurrent request handling
✅ **Resume Support**: Automatically skips already-generated files

---

## Installation

Make sure you have the dependencies:

```bash
pip install aiohttp tqdm datasets
```

Or use `uv` (recommended):

```bash
# The script has inline dependency metadata, so uv will handle it automatically
chmod +x custom_generate_completions.py
```

---

## Basic Usage

### Non-Streaming Mode (Default)

```bash
./custom_generate_completions.py \
    --api-url "http://onlineservice.cn-southwest-2:8087/csb-inner-service/rest/infers/cfd8c66f-3411-40c5-bf23-656926e30b00?path=/v1/chat/completions" \
    --model "pangu_auto" \
    --csb-token "253dfa9b-7fad35f" \
    --auth-token "nokey" \
    --output-dir ./results_no_stream \
    --completion-limit 20 \
    --batch-size 10
```

### Streaming Mode

```bash
./custom_generate_completions.py \
    --api-url "http://onlineservice.cn-southwest-2:8087/csb-inner-service/rest/infers/cfd8c66f-3411-40c5-bf23-656926e30b00?path=/v1/chat/completions" \
    --model "pangu_auto" \
    --csb-token "253dfa9b-7fad35f" \
    --auth-token "nokey" \
    --stream \
    --output-dir ./results_stream \
    --completion-limit 20 \
    --batch-size 10
```

### One-Shot Prompting (Better Performance)

```bash
./custom_generate_completions.py \
    --api-url "http://onlineservice.cn-southwest-2:8087/csb-inner-service/rest/infers/cfd8c66f-3411-40c5-bf23-656926e30b00?path=/v1/chat/completions" \
    --model "pangu_auto" \
    --csb-token "253dfa9b-7fad35f" \
    --one-shot \
    --output-dir ./results_oneshot \
    --completion-limit 20
```

---

## All Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--api-url` | ✅ Yes | - | Full API endpoint URL (including query params) |
| `--model` | No | `pangu_auto` | Model name to send in request |
| `--csb-token` | ✅ Yes | - | CSB authentication token |
| `--auth-token` | No | `nokey` | Bearer token for Authorization header |
| `--stream` | No | `False` | Enable streaming mode (flag) |
| `--one-shot` | No | `False` | Use one-shot prompting (flag) |
| `--output-dir` | ✅ Yes | - | Directory to save results |
| `--batch-size` | No | `10` | Number of concurrent API requests |
| `--completion-limit` | No | `20` | Completions per prompt (for Pass@k) |
| `--temperature` | No | `0.7` | Sampling temperature |
| `--top-p` | No | `0.8` | Top-p (nucleus) sampling |
| `--top-k` | No | `20` | Top-k sampling |
| `--max-tokens` | No | `3072` | Maximum tokens per completion |
| `--timeout` | No | `60` | Timeout per request (seconds) |
| `--dataset` | No | `nuprl/CanItEdit` | HuggingFace dataset to use |
| `--split` | No | `test` | Dataset split |

---

## Example: Quick Test (1 completion per prompt)

For prototyping, use `--completion-limit 1`:

```bash
./custom_generate_completions.py \
    --api-url "http://onlineservice.cn-southwest-2:8087/csb-inner-service/rest/infers/cfd8c66f-3411-40c5-bf23-656926e30b00?path=/v1/chat/completions" \
    --csb-token "253dfa9b-7fad35f" \
    --output-dir ./test_run \
    --completion-limit 1 \
    --batch-size 5
```

This generates only 1 completion per example, completing much faster (~210 API calls instead of 4200).

---

## Output Format

The script generates `.json.gz` files in the output directory:

```
output_dir/
├── problem_1_instruction_descriptive.json.gz
├── problem_1_instruction_lazy.json.gz
├── problem_2_instruction_descriptive.json.gz
├── ...
```

Each file contains:
- Original example metadata (`before`, `after`, `test`, etc.)
- Generated completions array
- Model parameters used
- Script arguments

---

## Next Steps: Evaluation

After generating completions, evaluate them:

```bash
# 1. Run tests and calculate coverage
./evaluate_completions.sh ./results_stream

# 2. Calculate Pass@k and ExcessCode metrics
./pass_k.py ./results_stream
```

---

## Comparing Streaming vs Non-Streaming

Generate both and compare:

```bash
# Non-streaming
./custom_generate_completions.py \
    --api-url "..." \
    --csb-token "..." \
    --output-dir ./results_no_stream \
    --completion-limit 20

# Streaming
./custom_generate_completions.py \
    --api-url "..." \
    --csb-token "..." \
    --stream \
    --output-dir ./results_stream \
    --completion-limit 20

# Evaluate both
./evaluate_completions.sh ./results_no_stream
./evaluate_completions.sh ./results_stream

# Compare metrics
./pass_k.py ./results_no_stream ./results_stream | column -t -s,
```

---

## Troubleshooting

### Connection Errors

If you get connection errors, check:
1. Is the API URL correct and reachable?
2. Is the `csb-token` valid?
3. Try reducing `--batch-size` to avoid rate limits

### Timeouts

If you get many `<timeout encountered>`, try:
1. Increase `--timeout` (default: 60s)
2. Reduce `--batch-size` to avoid overloading the server
3. Reduce `--max-tokens` for faster generation

### Testing Endpoint Connectivity

Quick Python test:

```python
import aiohttp
import asyncio
import json

async def test_endpoint():
    url = "http://onlineservice.cn-southwest-2:8087/csb-inner-service/rest/infers/cfd8c66f-3411-40c5-bf23-656926e30b00?path=/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer nokey",
        "csb-token": "253dfa9b-7fad35f",
    }
    payload = {
        "model": "pangu_auto",
        "messages": [{"role": "user", "content": "Hello, world!"}],
        "stream": False,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            print(f"Status: {resp.status}")
            result = await resp.json()
            print(json.dumps(result, indent=2))

asyncio.run(test_endpoint())
```

---

## Questions?

- Streaming not working? Make sure your endpoint supports SSE (Server-Sent Events)
- Need different prompt format? Modify `chat_edit_prompt_zeroshot()` function
- Want to add more parameters? Add them to the `payload` dict in `generate_non_streaming()`
