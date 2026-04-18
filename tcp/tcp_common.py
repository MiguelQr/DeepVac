import socket
import struct
from typing import Dict, List, Tuple

DEFAULT_HOST = "192.168.88.188"
DEFAULT_PORT = 4321
DEFAULT_TIMEOUT = 5.0

HEADER = b"\xAA\x55\xBB\x77"

# Text command
TEXT_CMD_CODE = 0x01

# App -> controller
OUTGOING_SETTINGS_MAP_CODE = 0x02
OUTGOING_JOB_CODE = 0x03

# Controller -> app
INCOMING_STATE_VALUES_CODE = 0x01
INCOMING_STATE_NAMES_CODE = 0x02
INCOMING_SETTINGS_MAP_CODE = 0x03
INCOMING_CONFIGURATION_CODE = 0x04
INCOMING_ALARM_CODE = 0x05
INCOMING_JOB_FINISHED_CODE = 0x06
INCOMING_USER_MSG_CODE = 0x07


def crc16_ccitt_false(data: bytes) -> int:
    poly = 0x1021
    crc = 0xFFFF

    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) & 0xFFFF) ^ poly
            else:
                crc = (crc << 1) & 0xFFFF

    return crc & 0xFFFF


def make_pascal_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    if len(raw) > 255:
        raise ValueError(f"String too long: {text}")
    return struct.pack("<B", len(raw)) + raw


def make_text_cmd_body(cmd: str) -> bytes:
    return bytes([TEXT_CMD_CODE]) + make_pascal_string(cmd)


def make_settings_map_body(settings: Dict[str, float], msg_code: int = OUTGOING_SETTINGS_MAP_CODE) -> bytes:
    if len(settings) > 255:
        raise ValueError("Too many settings")

    body = bytearray([msg_code, len(settings)])

    for key, value in settings.items():
        body += make_pascal_string(key)
        body += struct.pack("<f", float(value))

    return bytes(body)


def make_packet(body: bytes) -> bytes:
    packet_wo_header = bytearray()
    packet_wo_header += struct.pack("<H", len(body))
    packet_wo_header += body

    crc = crc16_ccitt_false(bytes(packet_wo_header))

    # Match StreamBuffer::makePackage(): CRC high byte first, then low byte
    packet_wo_header.append((crc >> 8) & 0xFF)
    packet_wo_header.append(crc & 0xFF)

    return HEADER + bytes(packet_wo_header)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < n:
        chunk = sock.recv(n - len(chunks))
        if not chunk:
            raise ConnectionError("Socket closed while reading data")
        chunks += chunk
    return bytes(chunks)


def read_one_packet(sock: socket.socket) -> bytes:
    header = recv_exact(sock, 4)
    if header != HEADER:
        raise ValueError(f"Bad header: {header.hex(' ')}")

    size_bytes = recv_exact(sock, 2)
    body_size = struct.unpack("<H", size_bytes)[0]

    body = recv_exact(sock, body_size)
    crc_bytes = recv_exact(sock, 2)

    crc_check = crc16_ccitt_false(size_bytes + body + crc_bytes)
    if crc_check != 0:
        raise ValueError(f"CRC check failed: 0x{crc_check:04X}")

    return body


def parse_pascal_string(data: bytes, offset: int) -> Tuple[str, int]:
    if offset >= len(data):
        raise ValueError("Offset out of range while parsing Pascal string")

    strlen = data[offset]
    offset += 1

    if offset + strlen > len(data):
        raise ValueError("Pascal string length exceeds available data")

    text = data[offset : offset + strlen].decode("utf-8", errors="replace")
    offset += strlen
    return text, offset


def parse_settings_map_body(
    body: bytes,
    expected_msg_code: int = INCOMING_SETTINGS_MAP_CODE,
    warn_trailing: bool = False,
) -> Dict[str, float]:
    if len(body) < 2:
        raise ValueError("Body too short")

    msg_code = body[0]
    if msg_code != expected_msg_code:
        raise ValueError(f"Unexpected message code: 0x{msg_code:02X}")

    count = body[1]
    offset = 2
    settings: Dict[str, float] = {}

    for _ in range(count):
        key, offset = parse_pascal_string(body, offset)

        if offset + 4 > len(body):
            raise ValueError("Not enough bytes for float value")

        value = struct.unpack("<f", body[offset : offset + 4])[0]
        offset += 4
        settings[key] = value

    if warn_trailing and offset != len(body):
        print(f"Warning: {len(body) - offset} trailing bytes left after parsing")

    return settings


def parse_state_names(body: bytes, expected_msg_code: int = INCOMING_STATE_NAMES_CODE) -> List[str]:
    if len(body) < 2 or body[0] != expected_msg_code:
        raise ValueError("Not a STATE_NAMES packet")

    count = body[1]
    offset = 2
    names: List[str] = []

    for _ in range(count):
        name, offset = parse_pascal_string(body, offset)
        names.append(name)

    return names


def parse_state_values(body: bytes, expected_msg_code: int = INCOMING_STATE_VALUES_CODE) -> List[float]:
    if len(body) < 2 or body[0] != expected_msg_code:
        raise ValueError("Not a STATE_VALUES packet")

    count = body[1]
    raw = body[2:]

    expected = count * 4
    if len(raw) < expected:
        raise ValueError("STATE_VALUES payload too short")

    return list(struct.unpack("<" + "f" * count, raw[:expected]))


def send_and_wait(
    body: bytes,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    timeout: float = DEFAULT_TIMEOUT,
) -> bytes:
    packet = make_packet(body)

    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(packet)
        return read_one_packet(sock)


def send_no_wait(
    body: bytes,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    timeout: float = DEFAULT_TIMEOUT,
) -> None:
    packet = make_packet(body)

    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(packet)


def request_settings(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    timeout: float = DEFAULT_TIMEOUT,
    expected_msg_code: int = INCOMING_SETTINGS_MAP_CODE,
) -> Dict[str, float]:
    body = make_text_cmd_body("get_settings")
    reply_body = send_and_wait(body, host=host, port=port, timeout=timeout)
    return parse_settings_map_body(reply_body, expected_msg_code=expected_msg_code)


def request_states(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, float]:
    packet = make_packet(make_text_cmd_body("get_states"))

    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(packet)

        names = None
        values = None

        while names is None or values is None:
            body = read_one_packet(sock)
            msg_code = body[0]

            if msg_code == INCOMING_STATE_NAMES_CODE:
                names = parse_state_names(body)
            elif msg_code == INCOMING_STATE_VALUES_CODE:
                values = parse_state_values(body)

        if len(names) != len(values):
            raise ValueError(
                f"State names count ({len(names)}) does not match state values count ({len(values)})"
            )

        return dict(zip(names, values))