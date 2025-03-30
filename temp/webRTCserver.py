import asyncio
import websockets
import wave
import datetime
import os


class WebSocketAudioReceiver:
    def __init__(self, host='localhost', port=8080):
        self.host = host
        self.port = port
        self.recording = False
        self.audio_chunks = []
        self.current_wav_file = None

    async def start_server(self):
        async with websockets.serve(self.handle_client, self.host, self.port):
            print(f"WebSocket Audio Server started on {self.host}:{self.port}")
            # Keep server running forever
            await asyncio.Future()  # Run forever (Python 3.7+)

    async def handle_client(self, websocket, path=None):
        """
        For each new client connection, we continually listen for text or binary data.
        Text data is used as control messages (START_RECORDING, STOP_RECORDING).
        Binary data is appended as audio samples if we're currently recording.
        """
        print(f"Client connected: {websocket.remote_address}")
        try:
            while True:
                message = await websocket.recv()

                # Distinguish text commands from binary audio data
                if isinstance(message, str):
                    # This is a text command
                    self.handle_text_command(message)
                elif isinstance(message, bytes):
                    # This is audio data
                    self.process_audio_chunk(message)

        except websockets.ConnectionClosed:
            print("Client disconnected")
        except Exception as e:
            print(f"Error in handle_client: {e}")
        finally:
            # When the client disconnects, ensure we stop recording cleanly
            if self.recording:
                self.stop_recording()

    def handle_text_command(self, command):
        command = command.strip().upper()
        print(f"Received command: {command}")

        if command == "START_RECORDING":
            self.start_recording()
        elif command == "STOP_RECORDING":
            self.stop_recording()
        else:
            print("Unknown command received")

    def start_recording(self):
        if self.recording:
            print("Recording is already in progress.")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs('recordings', exist_ok=True)
        self.current_wav_file = f'recordings/audio_{timestamp}.wav'
        self.audio_chunks = []
        self.recording = True
        print(f"Started recording to {self.current_wav_file}")

    def process_audio_chunk(self, chunk):
        if self.recording:
            self.audio_chunks.append(chunk)

    def stop_recording(self):
        if not self.recording:
            print("No recording in progress to stop.")
            return

        self.save_wav_file()
        self.recording = False
        print(f"Stopped recording and saved to {self.current_wav_file}")
        self.current_wav_file = None
        self.audio_chunks = []

    def save_wav_file(self):
        if not self.audio_chunks:
            print("No audio data collected; nothing to save.")
            return

        with wave.open(self.current_wav_file, 'wb') as wav_file:
            wav_file.setnchannels(1)      # mono
            wav_file.setsampwidth(2)     # 16-bit
            wav_file.setframerate(44100)  # 44.1 kHz
            for chunk in self.audio_chunks:
                wav_file.writeframes(chunk)


async def main():
    receiver = WebSocketAudioReceiver()
    await receiver.start_server()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server shutdown via KeyboardInterrupt")
