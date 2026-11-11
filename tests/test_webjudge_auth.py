import httpx2
import pytest

from core.reward_models import webjudge


def test_token_file_is_read_for_every_request(tmp_path):
    token = tmp_path / "token"
    token.write_text("first\n")
    auth = webjudge._TokenFileAuth(str(token))

    request = auth(httpx2.Request("POST", "https://openrouter.ai/api/v1/x"))
    assert request.headers["Authorization"] == "Bearer first"

    token.write_text("second")
    request = auth(httpx2.Request("POST", "https://openrouter.ai/api/v1/x"))
    assert request.headers["Authorization"] == "Bearer second"


@pytest.mark.parametrize(
    ("api_key", "env_key", "expect_federated"),
    [("explicit", None, False), (None, "from-env", False), (None, None, True)],
)
def test_client_prefers_a_key_over_the_token_file(
    tmp_path, monkeypatch, api_key, env_key, expect_federated
):
    token = tmp_path / "token"
    token.write_text("access")
    monkeypatch.setenv("OPENROUTER_ACCESS_TOKEN_FILE", str(token))
    if env_key:
        monkeypatch.setenv("OPENROUTER_API_KEY", env_key)
    else:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    client = webjudge._openrouter_client(api_key, "https://openrouter.ai/api/v1")

    assert (client.api_key == "federated") == expect_federated
    if not expect_federated:
        assert client.api_key == (api_key or env_key)


async def test_federated_client_sends_the_current_token(tmp_path, monkeypatch):
    token = tmp_path / "token"
    token.write_text("access-1")
    monkeypatch.setenv("OPENROUTER_ACCESS_TOKEN_FILE", str(token))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    seen = []

    def handler(request):
        seen.append(request.headers["Authorization"])
        return httpx2.Response(200, json={"object": "list", "data": []})

    client = webjudge._openrouter_client(None, "https://openrouter.ai/api/v1")
    client._client._transport = httpx2.MockTransport(handler)
    await client.models.list()
    token.write_text("access-2")
    await client.models.list()
    assert seen == ["Bearer access-1", "Bearer access-2"]
