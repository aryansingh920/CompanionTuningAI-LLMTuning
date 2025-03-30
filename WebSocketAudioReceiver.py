import os
import wave
import asyncio
import websockets

class WebSocketAudioReceiver:
    """
    This class starts a WebSocket server that listens for audio from Unity.
    We save each START/STOP sequence as a new .wav file named 1.wav, 2.wav, 3.wav, etc.
    """

    def __init__(self, host='localhost', port=8080):
        self.host = host
        self.port = port
        self.recording = False
        self.audio_chunks = []
        self.current_wav_file = None

        # We'll store recordings in ./input
        os.makedirs('input', exist_ok=True)

        # Keep track of how many recordings we have so far (for naming 1.wav, 2.wav, etc.)
        self.file_counter = 0

    async def start_server(self):
        async with websockets.serve(self.handle_client, self.host, self.port):
            print(f"WebSocket Audio Server started on {self.host}:{self.port}")
            await asyncio.Future()  # Run forever

    async def handle_client(self, websocket, path=None):
        """
        For each new client connection, we continually listen for text or binary data.
        Text data is used as control messages: START_RECORDING / STOP_RECORDING
        Binary data is appended to self.audio_chunks if we're currently recording.
        """
        print(f"Client connected: {websocket.remote_address}")
        try:
            while True:
                message = await websocket.recv()
                if isinstance(message, str):
                    # It's a text command
                    self.handle_text_command(message)
                elif isinstance(message, bytes):
                    # It's audio data
                    self.process_audio_chunk(message)
        except websockets.ConnectionClosed:
            print("Client disconnected")
        except Exception as e:
            print(f"Error in handle_client: {e}")
        finally:
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
            print("Recording already in progress.")
            return
        # Bump the file counter for the new file
        self.file_counter += 1
        # e.g. input/1.wav, input/2.wav, etc.
        self.current_wav_file = f'input/{self.file_counter}.wav'
        self.audio_chunks = []
        self.recording = True
        print(f"Started recording -> {self.current_wav_file}")

    def process_audio_chunk(self, chunk):
        if self.recording:
            self.audio_chunks.append(chunk)

    def stop_recording(self):
        if not self.recording:
            print("No recording in progress; cannot stop.")
            return
        self.save_wav_file()
        self.recording = False
        print(f"Stopped recording; saved -> {self.current_wav_file}")
        self.current_wav_file = None
        self.audio_chunks = []

    def save_wav_file(self):
        if not self.audio_chunks:
            print("No audio data collected; nothing to save.")
            return
        with wave.open(self.current_wav_file, 'wb') as wav_file:
            wav_file.setnchannels(1)       # mono
            wav_file.setsampwidth(2)      # 16-bit
            wav_file.setframerate(44100)  # 44.1 kHz
            for chunk in self.audio_chunks:
                wav_file.writeframes(chunk)
