"""The audio domain: FastAPI, the WebSocket endpoint, and the Pipecat wiring.

`app/pipeline.py` is the only module in the repo that imports pipecat.
Everything about the conversation lives in `harness/`, which cannot import
anything here — enforced by tests/test_architecture.py.
"""
