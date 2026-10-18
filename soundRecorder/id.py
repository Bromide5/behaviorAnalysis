"""列出音频设备。

注意:必须在 import sounddevice 之前设置 SD_ENABLE_ASIO,
否则 sounddevice 会加载不带 ASIO 的 PortAudio DLL,ASIO 设备会完全消失。
"""

import os

os.environ.setdefault("SD_ENABLE_ASIO", "1")

import sounddevice as sd  # noqa: E402


def main() -> None:
    for i, h in enumerate(sd.query_hostapis()):
        print(f"[{i}] {h['name']}: {len(h['devices'])} 个设备")

    print()
    print(sd.query_devices())

    asio = [d for d in sd.query_devices() if "ASIO" in sd.query_hostapis()[d["hostapi"]]["name"]]
    if asio:
        print("\nASIO 设备:")
        for d in asio:
            print(f"  {d['name']} (in={d['max_input_channels']}, out={d['max_output_channels']})")
    else:
        print("\n没有枚举到 ASIO 设备")


if __name__ == "__main__":
    main()
