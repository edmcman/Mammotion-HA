"""Capture a snapshot frame from the Agora WebRTC stream using aiortc."""

from __future__ import annotations

import asyncio
import io
import logging
import re
import secrets
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from .coordinator import MammotionBaseUpdateCoordinator

_LOGGER = logging.getLogger(__name__)

_TRACK_TIMEOUT = 5
_FRAME_TIMEOUT = 7
_ICE_GATHER_TIMEOUT = 3
_SNAPSHOT_LOCK = asyncio.Lock()
_CANDIDATE_RE = re.compile(r"^a=(candidate:.+)$", re.MULTILINE)

# HEVC NAL unit types (RFC 7798)
_HEVC_NAL_AP = 48
_HEVC_NAL_FU = 49


def _extract_candidates_from_sdp(sdp: str) -> list[dict[str, Any]]:
    """Extract ICE candidates from SDP and convert to ORTC format."""
    candidates = []
    for match in _CANDIDATE_RE.finditer(sdp):
        cand_str = match.group(1)
        if cand_str.startswith("candidate:"):
            cand_str = cand_str[10:]
        parts = cand_str.split()
        if len(parts) < 8:
            continue
        try:
            candidates.append(
                {
                    "foundation": parts[0],
                    "ip": parts[4],
                    "port": int(parts[5]),
                    "priority": int(parts[3]),
                    "protocol": parts[2],
                    "type": parts[7],
                }
            )
        except (ValueError, IndexError):
            continue
    return candidates


def _depacketize_hevc_rtp(packets: list[tuple[int, bytes]]) -> bytes:
    """Depacketize HEVC RTP payloads into Annex B bitstream (RFC 7798).

    Takes a sorted list of (sequence_number, rtp_payload) for one frame
    and returns an Annex B HEVC bitstream ready for decoding.
    """
    annexb = bytearray()
    fu_buf = bytearray()
    fu_nal_hdr: bytes | None = None
    start_code = b"\x00\x00\x00\x01"

    for _seq, payload in packets:
        if len(payload) < 2:
            continue
        nal_type = (payload[0] >> 1) & 0x3F

        if nal_type == _HEVC_NAL_AP:
            # Aggregation packet: [NAL hdr 2B] [size 2B | NALU]...
            off = 2
            while off + 2 <= len(payload):
                nalu_size = (payload[off] << 8) | payload[off + 1]
                off += 2
                if off + nalu_size > len(payload):
                    break
                annexb += start_code + payload[off : off + nalu_size]
                off += nalu_size

        elif nal_type == _HEVC_NAL_FU:
            # Fragmentation unit: [NAL hdr 2B] [FU hdr 1B] [data...]
            if len(payload) < 3:
                continue
            fu_hdr = payload[2]
            s_bit = (fu_hdr >> 7) & 1
            e_bit = (fu_hdr >> 6) & 1
            fu_type = fu_hdr & 0x3F

            if s_bit:
                fu_nal_hdr = bytes(
                    [(payload[0] & 0x81) | (fu_type << 1), payload[1]]
                )
                fu_buf = bytearray(payload[3:])
            else:
                fu_buf += payload[3:]

            if e_bit and fu_nal_hdr is not None:
                annexb += start_code + fu_nal_hdr + bytes(fu_buf)
                fu_buf = bytearray()
                fu_nal_hdr = None

        else:
            # Single NAL unit packet
            annexb += start_code + payload

    return bytes(annexb)


def _hevc_annexb_to_jpeg(annexb_data: bytes) -> bytes | None:
    """Decode a single HEVC Annex B frame to JPEG using PyAV."""
    import av  # noqa: PLC0415

    codec = av.CodecContext.create("hevc", "r")
    packet = av.Packet(annexb_data)
    try:
        frames = codec.decode(packet)
    except av.error.InvalidDataError:
        _LOGGER.debug("Snapshot: HEVC decode returned InvalidDataError")
        return None

    if not frames:
        frames = codec.decode(None)

    if not frames:
        return None

    frame = frames[0]
    try:
        from PIL import Image  # noqa: PLC0415

        img: Image.Image = frame.to_image()
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except ImportError:
        pass

    # Fallback: encode via av
    yuv = frame.reformat(format="yuvj420p")
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="image2") as out:
        s = out.add_stream("mjpeg", rate=1)
        s.width = yuv.width
        s.height = yuv.height
        s.pix_fmt = "yuvj420p"
        for pkt in s.encode(yuv):
            buf.write(bytes(pkt))
    data = buf.getvalue()
    return data if data else None


_SNAPSHOT_BUDGET = 9.0  # seconds — must be less than HA's camera proxy timeout (10s)


async def capture_agora_snapshot(
    hass: HomeAssistant,
    coordinator: MammotionBaseUpdateCoordinator,
    width: int | None = None,
    height: int | None = None,
) -> bytes | None:
    """Capture a single JPEG frame from the Agora WebRTC stream.

    Enforces a 9-second overall budget so the result always arrives before
    Home Assistant's 10-second camera proxy deadline.
    """
    try:
        return await asyncio.wait_for(
            _capture_impl(hass, coordinator, width, height),
            timeout=_SNAPSHOT_BUDGET,
        )
    except asyncio.TimeoutError:
        _LOGGER.warning("Snapshot: overall budget exceeded (%.0fs)", _SNAPSHOT_BUDGET)
        return None


async def _capture_impl(
    hass: HomeAssistant,
    coordinator: MammotionBaseUpdateCoordinator,
    width: int | None = None,
    height: int | None = None,
) -> bytes | None:
    """Internal implementation — call via capture_agora_snapshot."""
    try:
        from aiortc import (  # noqa: PLC0415
            RTCConfiguration,
            RTCIceCandidate,
            RTCIceServer,
            RTCPeerConnection,
            RTCSessionDescription,
        )
    except ImportError:
        _LOGGER.error("aiortc is not installed")
        return None

    async with _SNAPSHOT_LOCK:
        stream_data, agora_response = await coordinator.async_check_stream_expiry()
        if not stream_data or stream_data.data is None:
            _LOGGER.warning("Snapshot: no stream data available")
            return None

        stream_was_inactive = coordinator._active_webrtc_sessions <= 0
        if stream_was_inactive:
            await coordinator.join_webrtc_channel()

        ice_servers: list[RTCIceServer] = []
        if agora_response:
            ice_servers = [
                RTCIceServer(urls=s.urls, username=s.username, credential=s.credential)
                for s in agora_response.get_ice_servers(use_all_turn_servers=False)
            ]

        config = RTCConfiguration(iceServers=ice_servers) if ice_servers else None
        pc = RTCPeerConnection(configuration=config)

        video_track_event: asyncio.Event = asyncio.Event()
        conn_event: asyncio.Event = asyncio.Event()

        @pc.on("track")
        def on_track(track: Any) -> None:
            if track.kind == "video":
                video_track_event.set()

        @pc.on("connectionstatechange")
        def on_conn_state() -> None:
            _LOGGER.debug("Snapshot: connection state -> %s", pc.connectionState)
            if pc.connectionState == "connected":
                conn_event.set()

        handler = None
        try:
            pc.addTransceiver("video", direction="recvonly")
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)
            await _wait_for_ice_gathering(pc)

            offer_sdp = pc.localDescription.sdp
            our_candidates = _extract_candidates_from_sdp(offer_sdp)

            from .agora_websocket import AgoraWebSocketHandler  # noqa: PLC0415
            from webrtc_models import RTCIceCandidateInit  # noqa: PLC0415

            handler = AgoraWebSocketHandler(hass)
            for cand in our_candidates:
                cand_line = (
                    f"candidate:{cand['foundation']} 1 {cand['protocol']} "
                    f"{cand['priority']} {cand['ip']} {cand['port']} "
                    f"typ {cand['type']}"
                )
                handler.candidates.append(
                    RTCIceCandidateInit(
                        candidate=cand_line, sdp_mid="0", sdp_m_line_index=0
                    )
                )

            answer_sdp = await handler.connect_and_join(
                stream_data.data,
                offer_sdp,
                secrets.token_hex(8),
                agora_response,
            )

            if not answer_sdp:
                _LOGGER.error("Snapshot: no answer SDP received from Agora")
                return None

            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=answer_sdp, type="answer")
            )

            if agora_response:
                for addr in agora_response.get_gateway_addresses():
                    try:
                        await pc.addIceCandidate(
                            RTCIceCandidate(
                                component=1,
                                foundation="agora",
                                ip=addr.ip,
                                port=addr.port,
                                priority=2130706431,
                                protocol="udp",
                                type="host",
                                sdpMid="0",
                                sdpMLineIndex=0,
                            )
                        )
                    except Exception:
                        pass

            await asyncio.wait_for(video_track_event.wait(), timeout=_TRACK_TIMEOUT)
            await asyncio.wait_for(conn_event.wait(), timeout=_TRACK_TIMEOUT)

            # Inject PT=0 routing — the Agora gateway uses PT 0 instead of the
            # negotiated PT, so we register it manually with the RTP router.
            receiver = pc.getReceivers()[0] if pc.getReceivers() else None
            dtls_transport = receiver.transport if receiver else None
            if dtls_transport and receiver:
                router = dtls_transport._rtp_router
                if 0 not in router.payload_type_table:
                    router.payload_type_table[0] = set()
                router.payload_type_table[0].add(receiver)
                router.ssrc_table[40000] = receiver

            # Intercept RTP at the DTLS layer, collect one HEVC keyframe,
            # decode it, and return JPEG.
            frame_ready: asyncio.Event = asyncio.Event()
            frame_packets: dict[int, list[tuple[int, bytes]]] = {}
            jpeg_result: list[bytes | None] = [None]
            got_keyframe_start = [False]

            if dtls_transport:
                orig_handle = dtls_transport._handle_rtp_data

                async def _intercept_rtp(
                    data: bytes,
                    arrival_time_ms: float,
                    _orig: Any = orig_handle,
                ) -> None:
                    from aiortc.rtp import RtpPacket  # noqa: PLC0415

                    if not frame_ready.is_set():
                        try:
                            pkt = RtpPacket.parse(
                                data, dtls_transport._rtp_header_extensions_map
                            )
                            ts = pkt.timestamp
                            payload = pkt.payload
                            if len(payload) >= 2:
                                nal_type = (payload[0] >> 1) & 0x3F
                                is_keyframe_start = False
                                if nal_type == _HEVC_NAL_AP:
                                    is_keyframe_start = True
                                elif nal_type == _HEVC_NAL_FU and len(payload) >= 3:
                                    fu_hdr = payload[2]
                                    fu_type = fu_hdr & 0x3F
                                    s_bit = (fu_hdr >> 7) & 1
                                    if s_bit and fu_type in (19, 20):
                                        is_keyframe_start = True

                                if is_keyframe_start:
                                    got_keyframe_start[0] = True

                                if got_keyframe_start[0]:
                                    if ts not in frame_packets:
                                        frame_packets[ts] = []
                                    frame_packets[ts].append(
                                        (pkt.sequence_number, bytes(payload))
                                    )

                                # Decode when we see the next timestamp
                                if (
                                    got_keyframe_start[0]
                                    and len(frame_packets) >= 2
                                    and not frame_ready.is_set()
                                ):
                                    first_ts = min(frame_packets)
                                    pkts = sorted(
                                        frame_packets[first_ts], key=lambda x: x[0]
                                    )
                                    annexb = _depacketize_hevc_rtp(pkts)
                                    if annexb:
                                        jpeg_result[0] = (
                                            await hass.async_add_executor_job(
                                                _hevc_annexb_to_jpeg, annexb
                                            )
                                        )
                                        if jpeg_result[0]:
                                            _LOGGER.info(
                                                "Snapshot: decoded HEVC keyframe "
                                                "(%d packets, %d bytes JPEG)",
                                                len(pkts),
                                                len(jpeg_result[0]),
                                            )
                                    frame_ready.set()
                        except Exception:
                            _LOGGER.debug(
                                "Snapshot: RTP intercept error", exc_info=True
                            )

                    await _orig(data, arrival_time_ms)

                dtls_transport._handle_rtp_data = _intercept_rtp

            # Send PLI immediately, then retry every 2 seconds until keyframe
            # arrives or timeout expires. A single PLI is often dropped or the
            # mower's IDR interval exceeds the wait window.
            async def _pli_loop() -> None:
                while not frame_ready.is_set():
                    if receiver:
                        try:
                            await receiver._send_rtcp_pli(40000)
                            _LOGGER.debug("Snapshot: sent PLI requesting keyframe")
                        except Exception:
                            pass
                    await asyncio.sleep(1)

            pli_task = asyncio.ensure_future(_pli_loop())
            try:
                await asyncio.wait_for(frame_ready.wait(), timeout=_FRAME_TIMEOUT)
            except asyncio.TimeoutError:
                _LOGGER.warning("Snapshot: timed out waiting for HEVC keyframe")
            finally:
                pli_task.cancel()

            return jpeg_result[0]

        except Exception:
            _LOGGER.exception("Snapshot: unexpected error during capture")
            return None

        finally:
            await pc.close()
            if handler is not None:
                await handler.disconnect()
            if stream_was_inactive:
                await coordinator.leave_webrtc_channel()


async def _wait_for_ice_gathering(pc: Any) -> None:
    """Wait until ICE gathering is complete, or until the timeout elapses."""
    if pc.iceGatheringState == "complete":
        return

    done: asyncio.Event = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def _on_change() -> None:
        if pc.iceGatheringState == "complete":
            done.set()

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
