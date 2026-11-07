"""RETROFIT AI operator-assist engine: healthy-twin residuals + simulator-trained diagnosis + safe alert policy.

Read-only by design: the engine consumes telemetry and context and emits advisory alerts. It has no write path to
any device, controller or safety system.
"""
__version__ = "0.1.0"
