from subscriber.kvcm.manager_client import HttpKvCacheManagerClient


async def test_http_manager_client_has_awaitable_readiness() -> None:
    client = HttpKvCacheManagerClient("http://127.0.0.1:8080")

    try:
        assert await client.is_ready() is True
    finally:
        await client.close()
