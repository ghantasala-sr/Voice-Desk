"""Tools: the registry, the idempotency ledger and executor, and read-back.

The claim this package exists to support is ZERO duplicate side effects under
interruption. That claim is only interesting if it is hard to satisfy, which
is why the mocks are adversarial (mocks/faults.py) and why the enforcement
point is a database constraint rather than application logic (docs/07).
"""
