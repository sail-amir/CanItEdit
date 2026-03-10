#!/usr/bin/env python3
"""
Quick test script to verify your custom endpoint works before running the full benchmark.
"""

import aiohttp
import asyncio
import json
import sys


async def test_non_streaming(api_url: str, csb_token: str, auth_token: str = "nokey"):
    """Test non-streaming mode"""
    print("=" * 60)
    print("Testing NON-STREAMING mode")
    print("=" * 60)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}",
        "csb-token": csb_token,
    }

    payload = {
        "model": "pangu_auto",
        "messages": [
            {
                "role": "user",
                "content": "Write a simple Python function to add two numbers."
            }
        ],
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "max_tokens": 200,
        "stream": False,
    }

    print(f"\nAPI URL: {api_url}")
    print(f"CSB Token: {csb_token[:10]}...")
    print(f"\nRequest payload:")
    print(json.dumps(payload, indent=2))
    print("\nSending request...")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, json=payload, headers=headers) as resp:
                print(f"Status: {resp.status}")

                if resp.status != 200:
                    error_text = await resp.text()
                    print(f"❌ Error: {error_text}")
                    return False

                result = await resp.json()
                print("\n✅ Success! Response:")
                print(json.dumps(result, indent=2))

                # Extract and display the generated content
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    print("\n" + "=" * 60)
                    print("Generated Content:")
                    print("=" * 60)
                    print(content)
                    print("=" * 60)

                return True

    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_streaming(api_url: str, csb_token: str, auth_token: str = "nokey"):
    """Test streaming mode"""
    print("\n\n")
    print("=" * 60)
    print("Testing STREAMING mode")
    print("=" * 60)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}",
        "csb-token": csb_token,
    }

    payload = {
        "model": "pangu_auto",
        "messages": [
            {
                "role": "user",
                "content": "Write a simple Python function to multiply two numbers."
            }
        ],
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "max_tokens": 200,
        "stream": True,
    }

    print(f"\nAPI URL: {api_url}")
    print(f"CSB Token: {csb_token[:10]}...")
    print(f"\nRequest payload:")
    print(json.dumps(payload, indent=2))
    print("\nSending streaming request...")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, json=payload, headers=headers) as resp:
                print(f"Status: {resp.status}")

                if resp.status != 200:
                    error_text = await resp.text()
                    print(f"❌ Error: {error_text}")
                    return False

                print("\n✅ Streaming chunks:")
                print("=" * 60)

                full_content = ""
                chunk_count = 0

                async for line in resp.content:
                    line = line.decode('utf-8').strip()
                    if not line:
                        continue

                    if line == "data: [DONE]":
                        print("\n[DONE]")
                        break

                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])  # Remove "data: " prefix
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                content_chunk = delta.get("content", "")
                                if content_chunk:
                                    full_content += content_chunk
                                    print(content_chunk, end="", flush=True)
                                    chunk_count += 1
                        except json.JSONDecodeError as e:
                            print(f"\n[JSON decode error: {e}]")
                            continue

                print("\n" + "=" * 60)
                print(f"\nReceived {chunk_count} chunks")
                print(f"Total content length: {len(full_content)} chars")
                print("=" * 60)

                return True

    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    if len(sys.argv) < 3:
        print("Usage: python test_custom_endpoint.py <API_URL> <CSB_TOKEN> [AUTH_TOKEN]")
        print("\nExample:")
        print('  python test_custom_endpoint.py \\')
        print('    "http://onlineservice.cn-southwest-2:8087/csb-inner-service/rest/infers/cfd8c66f-3411-40c5-bf23-656926e30b00?path=/v1/chat/completions" \\')
        print('    "253dfa9b-7fad35f" \\')
        print('    "nokey"')
        sys.exit(1)

    api_url = sys.argv[1]
    csb_token = sys.argv[2]
    auth_token = sys.argv[3] if len(sys.argv) > 3 else "nokey"

    # Test both modes
    result_non_streaming = await test_non_streaming(api_url, csb_token, auth_token)
    result_streaming = await test_streaming(api_url, csb_token, auth_token)

    # Summary
    print("\n\n")
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Non-streaming: {'✅ PASSED' if result_non_streaming else '❌ FAILED'}")
    print(f"Streaming:     {'✅ PASSED' if result_streaming else '❌ FAILED'}")
    print("=" * 60)

    if result_non_streaming and result_streaming:
        print("\n🎉 All tests passed! Your endpoint is ready to use.")
        print("\nNext step: Run the benchmark:")
        print("  ./custom_generate_completions.py \\")
        print(f'    --api-url "{api_url}" \\')
        print(f'    --csb-token "{csb_token}" \\')
        print('    --output-dir ./results \\')
        print('    --completion-limit 1  # Start with 1 for testing')
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
