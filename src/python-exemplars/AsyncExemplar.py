'''
Created on 11/8/2025 at 6:06 PM
By yuvaraj
Module Name: AsyncExemplar
'''
import asyncio

import aiohttp
from dotenv import load_dotenv

load_dotenv()

'''
Key Takeaways
---------------------------------------------------------------------------
Concept         |   Description
---------------------------------------------------------------------------
async def       | 	Defines a coroutine (an async function)
await           |	Pauses the coroutine until the awaited task completes
asyncio.run()   | 	Starts and runs the main event loop
asyncio.gather()|	Runs multiple coroutines concurrently
---------------------------------------------------------------------------
Best for	I/O-bound tasks (network, database, file I/O)
'''

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main():
    urls = [
        "https://example.com",
        "https://python.org",
        "https://httpbin.org/get"
    ]

    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        responses = await asyncio.gather(*tasks)
        for i, text in enumerate(responses):
            print(f"Response {i} length: {len(text)}")


if __name__ == '__main__':
    asyncio.run(main())
