#!/usr/bin/env python3
"""
Test script for SSE endpoint
"""

import asyncio
import aiohttp
import json

async def test_sse_endpoint():
    """Test the SSE endpoint"""
    url = "http://localhost:8000/sse"
    
    print(f"Connecting to SSE endpoint: {url}")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            print(f"Response status: {response.status}")
            print(f"Response headers: {dict(response.headers)}")
            
            # Read the SSE stream
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line:
                    print(f"Received: {line}")

async def test_api_endpoints():
    """Test other API endpoints"""
    base_url = "http://localhost:8000"
    
    async with aiohttp.ClientSession() as session:
        # Test root endpoint
        async with session.get(f"{base_url}/") as response:
            data = await response.json()
            print(f"Root endpoint: {data}")
        
        # Test status endpoint
        async with session.get(f"{base_url}/status") as response:
            data = await response.json()
            print(f"Status endpoint: {data}")

if __name__ == "__main__":
    print("Testing SSE endpoint...")
    try:
        asyncio.run(test_api_endpoints())
        print("\nTesting SSE stream (will run for 10 seconds)...")
        asyncio.run(asyncio.wait_for(test_sse_endpoint(), timeout=10))
    except asyncio.TimeoutError:
        print("SSE test completed (timeout)")
    except Exception as e:
        print(f"Error: {e}") 