"""The harness: everything about the conversation.

Pipecat owns audio. The harness owns state, LLM calls, retrieval, tools,
checks, stop rules and handoff. Nothing in this package imports pipecat, app,
or eval — enforced by an import-linter contract in pyproject.toml.

That boundary is what makes `eval/text_driver.py` possible: the entire
conversational system runs with typed input, deterministically, in
milliseconds, under pytest.
"""
