"""Capture a snapshot frame from the Agora WebRTC stream using aiortc."""

from __future__ import annotations

import asyncio
import io
import logging
import secrets
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from .coordinator import MammotionBaseUpdateCoordinator

_LOGGER = logging.getLogger(__name__)

# Seconds to wait for the first video track to appear after ICE/DTLS
_TRACK_TIMEOUT = 20
# Seconds to wait for the first decoded frame from the track
_FRAME_TIMEOUT = 30
# Seconds to wait for ICE gathering before proceeding with partial candidates
_ICE_GATHER_TIMEOUT = 8


async def capture_agora_snapshot(
    hass: HomeAssistant,
    coordinator: MammotionBaseUpdateCoordinator,
    width: int | None = None,
    height: int | None = None,
) -> bytes | None:
    """Capture a single JPEG frame from the Agora WebRTC stream.

    Opens a server-side WebRTC connection to the Agora channel using aiortc,
    receives the first video frame, encodes it as JPEG, then closes the
    connection.  Returns JPEG bytes or None on any failure.
    """
    try:
        from aiortc import (  # noqa: PLC0415
            RTCConfiguration,
            RTCIceServer,
            RTCPeerConnection,
            RTCSessionDescription,
        )
    except ImportError:
        _LOGGER.error(
            "aiortc is not installed; add 'aiortc' to requirements to enable "
            "camera snapshots"
        )
        return None

    # Fetch/refresh stream credentials
    stream_data, agora_response = await coordinator.async_check_stream_expiry()
    if not stream_data or stream_data.data is None:
        _LOGGER.warning("Snapshot: no stream data available")
        return None

    # Ask the device to start its WebRTC session
    await coordinator.join_webrtc_channel()

    # Build ICE server list from Agora TURN servers
    ice_servers: list[RTCIceServer] = []
    if agora_response:
        ice_servers = [
            RTCIceServer(urls=s.urls, username=s.username, credential=s.credential)
            for s in agora_response.get_ice_servers(use_all_turn_servers=False)
        ]

    config = RTCConfiguration(iceServers=ice_servers) if ice_servers else None
    pc = RTCPeerConnection(configuration=config)

    # Receive the video track via callback
    video_track_event: asyncio.Event = asyncio.Event()
    video_track_holder: list[Any] = []

    @pc.on("track")
    def on_track(track: Any) -> None:
        if track.kind == "video" and not video_track_event.is_set():
            video_track_holder.append(track)
            video_track_event.set()

    handler = None
    try:
        # Recvonly — we are a silent viewer, not a publisher
        pc.addTransceiver("video", direction="recvonly")
        pc.addTransceiver("audio", direction="recvonly")

        # Generate offer and set local description (starts ICE gathering)
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        # Wait for ICE gathering so the local description SDP is complete
        await _wait_for_ice_gathering(pc)

        # Send our offer through Agora's WebSocket signalling and get back answer
        from .agora_websocket import AgoraWebSocketHandler  # noqa: PLC0415

        handler = AgoraWebSocketHandler(hass)
        answer_sdp = await handler.connect_and_join(
            stream_data.data,
            pc.localDescription.sdp,
            secrets.token_hex(8),  # unique session id for this snapshot
            agora_response,
        )

        if not answer_sdp:
            _LOGGER.error("Snapshot: no answer SDP received from Agora")
            return None

        # Hand Agora's answer to our local peer connection to complete negotiation
        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer_sdp, type="answer")
        )

        # Wait for the video track to arrive (ICE + DTLS must complete first)
        try:
            await asyncio.wait_for(video_track_event.wait(), timeout=_TRACK_TIMEOUT)
        except asyncio.TimeoutError:
            _LOGGER.warning(
                "Snapshot: timed out waiting for video track "
                "(ICE/DTLS may not have completed)"
            )
            return None

        if not video_track_holder:
            return None

        # Pull one frame from the track
        video_track = video_track_holder[0]
        try:
            frame = await asyncio.wait_for(video_track.recv(), timeout=_FRAME_TIMEOUT)
        except asyncio.TimeoutError:
            _LOGGER.warning("Snapshot: timed out waiting for first video frame")
            return None

        # Encode in a thread so we don't block the event loop
        return await hass.async_add_executor_job(
            _frame_to_jpeg, frame, width, height
        )

    except Exception:
        _LOGGER.exception("Snapshot: unexpected error during capture")
        return None

    finally:
        await pc.close()
        if handler is not None:
            await handler.disconnect()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _wait_for_ice_gathering(pc: Any) -> None:
    """Wait until ICE gathering is complete, or until the timeout elapses."""
    if pc.iceGatheringState == "complete":
        return

    done: asyncio.Event = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def _on_change() -> None:
        if pc.iceGatheringState == "complete":
            done.set()

    # Re-check in case gathering completed between the if-check and handler setup
    if pc.iceGatheringState == "complete":
        done.set()

    try:
        await asyncio.wait_for(done.wait(), timeout=_ICE_GATHER_TIMEOUT)
    except asyncio.TimeoutError:
        _LOGGER.debug(
            "Snapshot: ICE gathering timed out after %ds, proceeding with "
            "partial candidates",
            _ICE_GATHER_TIMEOUT,
        )


def _frame_to_jpeg(
    frame: Any, width: int | None, height: int | None
) -> bytes | None:
    """Encode an av.VideoFrame to JPEG bytes (runs in a thread executor).

    Tries Pillow first (higher quality), falls back to a raw av encode.
    """
    # --- Pillow path ---
    try:
        from PIL import Image  # noqa: PLC0415

        img: Image.Image = frame.to_image()
        if width and height:
            img = img.resize((width, height), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception:  # noqa: BLE001
        pass  # fall through to av path

    # --- PyAV fallback (no Pillow needed) ---
    try:
        import av  # noqa: PLC0415

        # Reformat to the pixel format mjpeg expects
        yuv_frame = frame.reformat(format="yuv420p")
        if width and height:
            yuv_frame = yuv_frame.reformat(width=width, height=height)

        output = io.BytesIO()
        with av.open(output, mode="w", format="image2") as container:
            stream = container.add_stream("mjpeg", rate=1)
            stream.width = yuv_frame.width
            stream.height = yuv_frame.height
            stream.pix_fmt = "yuvj420p"
            for packet in stream.encode(yuv_frame):
                output.write(bytes(packet))
        data = output.getvalue()
        return data if data else None
    except Exception:
        _LOGGER.exception("Snapshot: JPEG encoding failed")
        return None
