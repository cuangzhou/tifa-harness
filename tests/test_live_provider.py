import os
import pytest

from tifa.providers import create_model_client


@pytest.mark.skipif(not os.getenv("TIFA_LIVE_TEST_PROVIDER"), reason="live provider test is opt-in")
def test_live_provider_read_only_smoke():
    provider = os.environ["TIFA_LIVE_TEST_PROVIDER"]
    client = create_model_client(provider, os.getenv("TIFA_LIVE_TEST_MODEL"))
    response = client.complete("Return a short final answer without tools.", [])
    assert response.text or response.tool_calls
