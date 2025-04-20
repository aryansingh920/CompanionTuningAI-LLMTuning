import os
import wave
import asyncio
import websockets


class WebSocketAudioReceiver:
    """
    WebSocket server that listens for audio from Unity and can stream audio data back.
    """

    def __init__(self, host='localhost', port=8080):
        self.host = host
        self.port = port
        self.recording = False
        self.audio_chunks = []
        self.current_wav_file = None
        self.connected_clients = set()  # Track all connected websockets

        os.makedirs('input', exist_ok=True)
        self.file_counter = 0

    async def start_server(self):
        async with websockets.serve(self.handle_client, self.host, self.port):
            print(f"WebSocket Audio Server started on {self.host}:{self.port}")
            await asyncio.Future()  # Run forever

    async def handle_client(self, websocket, path=None):
        """
        For each new client connection, store it in self.connected_clients,
        then continually listen for text or binary data.
        """
        print(f"Client connected: {websocket.remote_address}")
        self.connected_clients.add(websocket)
        try:
            while True:
                message = await websocket.recv()
                if isinstance(message, str):
                    self.handle_text_command(message)
                elif isinstance(message, bytes):
                    self.process_audio_chunk(message)
        except websockets.ConnectionClosed:
            print("Client disconnected")
        except Exception as e:
            print(f"Error in handle_client: {e}")
        finally:
            # Remove from connected set
            if websocket in self.connected_clients:
                self.connected_clients.remove(websocket)
            # If still recording, stop
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
        self.file_counter += 1
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

    async def broadcast_message(self, message: str):
        """
        Helper to send a text message to all connected clients.
        """
        disconnected = []
        for ws in self.connected_clients:
            try:
                await ws.send(message)
            except websockets.ConnectionClosed:
                disconnected.append(ws)
            except Exception as e:
                print(f"Error sending message to a client: {e}")
        # Remove any clients that disconnected
        for ws in disconnected:
            if ws in self.connected_clients:
                self.connected_clients.remove(ws)

    async def send_audio_data(self, audio_data: bytes, content_type="audio/mp3"):
        """
        Send raw audio bytes to all connected clients with a prefix
        indicating this is audio data and its format.
        """
        prefix = f"AUDIO_DATA:{content_type}:"
        prefix_bytes = prefix.encode('utf-8')

        # Combine prefix and audio data
        message = prefix_bytes + audio_data

        disconnected = []
        for ws in self.connected_clients:
            try:
                await ws.send(message)
                print(f"Sent {len(audio_data)} bytes of audio to client")
            except websockets.ConnectionClosed:
                disconnected.append(ws)
            except Exception as e:
                print(f"Error sending audio to a client: {e}")

        # Remove any clients that disconnected
        for ws in disconnected:
            if ws in self.connected_clients:
                self.connected_clients.remove(ws)
