import socket
import struct
from typing import Optional

from tcp_common import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    INCOMING_STATE_VALUES_CODE,
    OUTGOING_JOB_CODE,
    make_packet,
    parse_state_values,
    read_one_packet,
)

HOST = DEFAULT_HOST
PORT = DEFAULT_PORT
TIMEOUT = 10.0

PROGTYPE_SETPOINT = 1
TEMP_REF_INDEX = 2

# For a simple manual temp setpoint, use control_params defaults.
DEFAULT_FLAGS = 3


def make_one_interval_setpoint_job_body(temp_c: float, duration_s: int, flags: int) -> bytes:
    data = bytearray()

    data.append(OUTGOING_JOB_CODE)
    data.append(PROGTYPE_SETPOINT)
    data += struct.pack("<H", 0)  # intervalIndex
    data += struct.pack("<H", 0)  # cycle_count
    data += struct.pack("<H", 0)  # cycle_start
    data += struct.pack("<H", 0)  # cycle_end
    data += struct.pack("<H", 1)  # intervalsCount

    # control_params order: duration, temp, flags
    data += struct.pack("<I", int(duration_s))
    data += struct.pack("<f", float(temp_c))
    data += struct.pack("<I", int(flags))

    return bytes(data)


def read_temp_ref_once(sock: socket.socket, timeout_s: float = 15.0) -> Optional[float]:
    sock.settimeout(timeout_s)
    while True:
        body = read_one_packet(sock)
        if len(body) < 1 or body[0] != INCOMING_STATE_VALUES_CODE:
            continue
        values = parse_state_values(body)
        if TEMP_REF_INDEX < len(values):
            return values[TEMP_REF_INDEX]


def read_latest_temp_ref(sock: socket.socket, reads: int = 2, timeout_s: float = 5.0) -> Optional[float]:
    """
    Read a small number of subsequent STATE_VALUES packets and keep the latest temp_ref.
    This avoids trusting the very next packet after the write.
    """
    latest = None
    sock.settimeout(timeout_s)

    for _ in range(reads):
        body = read_one_packet(sock)
        if len(body) < 1 or body[0] != INCOMING_STATE_VALUES_CODE:
            continue
        values = parse_state_values(body)
        if TEMP_REF_INDEX < len(values):
            latest = values[TEMP_REF_INDEX]

    return latest


def main() -> None:
    new_temp = 24.0
    duration_s = 10
    flags = DEFAULT_FLAGS

    with socket.create_connection((HOST, PORT), timeout=TIMEOUT) as sock:
        sock.settimeout(TIMEOUT)

        before = read_temp_ref_once(sock, timeout_s=15.0)
        print("temp_ref before =", before)

        body = make_one_interval_setpoint_job_body(
            temp_c=new_temp,
            duration_s=duration_s,
            flags=flags,
        )
        sock.sendall(make_packet(body))
        print(f"Sent one-interval SETPOINT job: temp={new_temp}, duration={duration_s}, flags={flags}")

        after = read_latest_temp_ref(sock, reads=2, timeout_s=10.0)
        print("temp_ref after  =", after)

        if after is not None and abs(after - new_temp) < 1e-4:
            print("SUCCESS: temp_ref changed to requested value.")
        else:
            print("WARNING: temp_ref did not match the requested value.")


if __name__ == "__main__":
    main()