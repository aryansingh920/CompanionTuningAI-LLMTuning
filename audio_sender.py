import asyncio
import websockets
import sounddevice as sd
import numpy as np
from scipy import signal

WS_URI = "ws://localhost:8080"
SAMPLE_RATE = 44100
CHUNK_DURATION = 0.5  # seconds
CHANNELS = 1

# Noise reduction parameters
NOISE_REDUCTION_STRENGTH = 0.2
FILTER_ORDER = 2
LOW_CUT = 300  # Hz - reduce low frequency noise
HIGH_CUT = 3000  # Hz - focus on speech frequency range


async def stream_audio(ws, duration=5):
    loop = asyncio.get_running_loop()

    print("Starting mic stream with noise reduction...")

    # Design a bandpass filter to remove noise outside speech frequency range
    nyquist = 0.5 * SAMPLE_RATE
    low = LOW_CUT / nyquist
    high = HIGH_CUT / nyquist
    b, a = signal.butter(FILTER_ORDER, [low, high], btype='band')

    async def send_chunk(indata):
        try:
            # Apply noise reduction (simple amplitude threshold)
            filtered_data = signal.lfilter(b, a, indata.flatten())

            # Additional noise gate to reduce background noise
            amplitude = np.abs(filtered_data)
            mean_amplitude = np.mean(amplitude)
            noise_gate = NOISE_REDUCTION_STRENGTH * mean_amplitude
            filtered_data[amplitude < noise_gate] = 0

            # Reshape and convert to bytes
            filtered_data = filtered_data.reshape(indata.shape)
            await ws.send(filtered_data.tobytes())

        except Exception as e:
            print(f"Error sending audio chunk: {e}")

    def callback(indata, frames, time, status):
        if status:
            print(f"Stream status: {status}")
        asyncio.run_coroutine_threadsafe(send_chunk(indata), loop)

    # Start microphone stream with higher gain
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback, device=None):
        await asyncio.sleep(duration)


async def main():
    try:
        async with websockets.connect(WS_URI) as ws:
            print("Connected to server")

            while True:
                cmd = input(
                    "Press 'P' to record, 'T' to test audio levels, or 'Q' to quit: ").strip().lower()
                if cmd == 'q':
                    break
                elif cmd == 'p':
                    await ws.send("START_RECORDING")
                    print("Recording with noise reduction...")

                    await stream_audio(ws, duration=5)

                    await ws.send("STOP_RECORDING")
                    print("Recording sent.\n")
                elif cmd == 't':
                    print("Testing audio levels for 3 seconds...")
                    # Simple audio level test

                    def test_callback(indata, frames, time, status):
                        volume_norm = np.linalg.norm(indata) * 10
                        print(f"Input level: {volume_norm:.2f}")

                    with sd.InputStream(callback=test_callback,
                                        channels=CHANNELS,
                                        samplerate=SAMPLE_RATE):
                        await asyncio.sleep(3)
                else:
                    print(
                        "Unknown input. Use 'P' to record, 'T' to test audio levels, 'Q' to quit.")
    except websockets.exceptions.ConnectionRefusedError:
        print("Connection refused. Make sure the server is running at", WS_URI)


if __name__ == "__main__":
    try:
        # Check if scipy is installed
        try:
            import scipy
            print(f"Using scipy version {scipy.__version__}")
        except ImportError:
            print("Warning: scipy is not installed. Install with 'pip install scipy'")
            exit(1)

        # Check audio devices
        print("\nAvailable audio devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            print(
                f"{i}: {device['name']} (Inputs: {device['max_input_channels']}, Outputs: {device['max_output_channels']})")

        default_device = sd.query_devices(kind='input')
        print(f"\nDefault input device: {default_device['name']}")

        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped by user.")
