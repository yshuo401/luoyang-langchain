from langchain.agents import create_agent
from langchain_classic.chains import create_retrieval_chain
from langgraph.checkpoint.redis import RedisSaver


def test_high_level_langchain_api_is_available() -> None:
    assert callable(create_agent)
    assert callable(create_retrieval_chain)
    assert hasattr(RedisSaver, "setup")