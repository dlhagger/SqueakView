"""Durable operator-session supervision primitives."""

from .protocol import (
    MAX_FRAME_BYTES,
    PROTOCOL_VERSION,
    CommandEnvelope,
    EventEnvelope,
    FrameTooLargeError,
    HEARTBEAT_COMMAND_NAME,
    HEARTBEAT_REQUEST_ID,
    NewlineFrameDecoder,
    ProtocolError,
    SocketEnvelopeReader,
    SocketEnvelopeWriter,
    decode_envelope,
    encode_envelope,
    send_envelope,
)

__all__ = [
    "MAX_FRAME_BYTES",
    "PROTOCOL_VERSION",
    "CommandEnvelope",
    "EventEnvelope",
    "FrameTooLargeError",
    "HEARTBEAT_COMMAND_NAME",
    "HEARTBEAT_REQUEST_ID",
    "NewlineFrameDecoder",
    "ProtocolError",
    "SocketEnvelopeReader",
    "SocketEnvelopeWriter",
    "decode_envelope",
    "encode_envelope",
    "send_envelope",
]
