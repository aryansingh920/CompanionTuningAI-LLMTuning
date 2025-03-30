import os
import re
import wave
import queue
import torch
import whisper
import asyncio
import websockets
import datetime
import numpy as np
import soundfile as sf
from gtts import gTTS
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------------------------------------------------------------
# Therapy Chatbot class
# --------------------------------------------------------------------------------


class TherapyChatbot:
    def __init__(self, model_path="./fine_tuned_therapy_gpt"):
        # ------------------------------------------------------------------------
        # 1) Load Whisper for transcription
        # ------------------------------------------------------------------------
        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model("base")  # or "small", etc.

        # ------------------------------------------------------------------------
        # 2) Load fine-tuned GPT model for therapy-style response
        # ------------------------------------------------------------------------
        print("Loading therapy model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)

        # ------------------------------------------------------------------------
        # 3) Select device (MPS on Mac, CUDA on GPU, else CPU)
        # ------------------------------------------------------------------------
        self.device = torch.device("mps" if torch.backends.mps.is_available()
                                   else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Using device: {self.device}")
        self.model = self.model.to(self.device)

        # ------------------------------------------------------------------------
        # 4) Some fallback responses for any errors or safety checks
        # ------------------------------------------------------------------------
        self.fallback_responses = [
            "I'm listening. Can you tell me more about that?",
            "That sounds challenging. How has this been affecting you?",
            "I'm here to support you. What would be most helpful to discuss right now?",
            "Thank you for sharing. How long have you been feeling this way?",
            "I understand this is important to you. Could we explore that further?"
        ]
        self.fallback_index = 0

    def transcribe_audio(self, file_path):
        """
        Transcribe a WAV file using Whisper.
        """
        print(f"Transcribing {file_path}...")
        result = self.whisper_model.transcribe(file_path)
        return result['text'].strip()

    def generate_response(self, transcript):
        """
        Generate a response from the therapy model, given a user transcript.
        """
        try:
            # Build a prompt
            prompt = (
                "As a professional, empathetic therapist, provide helpful and ethical advice. "
                "Focus on understanding the person's concerns without making assumptions.\n"
                f"questionText: {transcript}\nanswerText: "
            )

            inputs = self.tokenizer(prompt, return_tensors="pt",
                                    padding=True, truncation=True).to(self.device)

            output = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=100,
                temperature=0.6,
                top_p=0.92,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                num_return_sequences=1,
                do_sample=True
            )

            response = self.tokenizer.decode(
                output[0], skip_special_tokens=True)
            # Remove the prompt portion from the raw decode (if necessary)
            response = response[len(prompt):].strip()

            # Clean up
            response = self.clean_response(response)
            return response

        except Exception as e:
            print(f"Error generating response: {e}")
            return self.get_fallback_response()

    def clean_response(self, response):
        """
        Basic cleanup/filters for the generated text.
        """
        # Remove repeated patterns
        response = re.sub(r'(.{20,}?)\1+', r'\1', response)

        # Filter certain words/phrases
        inappropriate_phrases = [
            "answerText:", "cheating", "questionText:"
        ]
        for phrase in inappropriate_phrases:
            if phrase in response:
                return self.get_fallback_response()

        # Make sure we have a reasonably-length answer
        return response if len(response) > 10 else self.get_fallback_response()

    def get_fallback_response(self):
        """
        Rotate through fallback responses.
        """
        resp = self.fallback_responses[self.fallback_index]
        self.fallback_index = (self.fallback_index +
                               1) % len(self.fallback_responses)
        return resp

    def text_to_speech(self, text, output_file):
        """
        Convert text to speech using gTTS, saving to an MP3.
        """
        try:
            tts = gTTS(text=text, lang="en", slow=False)
            tts.save(output_file)
            print(f"TTS saved to {output_file}")
        except Exception as e:
            print(f"Text-to-speech error: {e}")


# --------------------------------------------------------------------------------
# WebSocketAudioReceiver class
# --------------------------------------------------------------------------------

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


# --------------------------------------------------------------------------------
# Background task to watch the 'input' folder and process any new WAV files.
# We'll produce TTS output in 'output' folder with matching numeric names.
# --------------------------------------------------------------------------------

async def watch_and_process(chatbot: TherapyChatbot):
    """
    Continuously monitors the 'input' folder for any new .wav files named 1.wav, 2.wav, etc.
    Processes them in ascending order, transcribes, generates response, does TTS,
    and saves the TTS audio to output/<same_number>.mp3, then removes the input file.
    """
    os.makedirs('output', exist_ok=True)

    processed_files = set()  # Keep track of which files we've processed

    while True:
        all_wav = [f for f in os.listdir('input') if f.endswith('.wav')]
        # Filter out anything we've already processed
        unprocessed = [f for f in all_wav if f not in processed_files]

        # Sort them numerically by the part before .wav
        def numeric_key(filename):
            try:
                return int(os.path.splitext(filename)[0])
            except ValueError:
                return 999999  # If something weird is there

        unprocessed.sort(key=numeric_key)

        # For each unprocessed WAV file in ascending order
        for wav_file in unprocessed:
            input_path = os.path.join('input', wav_file)
            file_stem = os.path.splitext(wav_file)[0]
            output_path = os.path.join('output', f"{file_stem}.mp3")

            print(f"\n[PROCESSING] {input_path} -> {output_path}")

            # 1) Transcribe
            transcript = chatbot.transcribe_audio(input_path)
            print(f"Transcript: {transcript}")

            # 2) Generate therapy response
            response = chatbot.generate_response(transcript)
            print(f"Response: {response}")

            # 3) TTS
            chatbot.text_to_speech(response, output_path)

            # Mark as processed
            processed_files.add(wav_file)

            # 4) Delete the input WAV to "clear" it
            try:
                os.remove(input_path)
                print(f"Removed input file: {input_path}")
            except Exception as e:
                print(f"Error removing {input_path}: {e}")

        # Sleep briefly to avoid tight-looping
        await asyncio.sleep(2)


# --------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------

async def main():
    # Create the therapy chatbot
    chatbot = TherapyChatbot()

    # Start the audio receiver (WebSocket server)
    receiver = WebSocketAudioReceiver(host='localhost', port=8080)

    # Launch two tasks concurrently:
    #   1) The WebSocket server
    #   2) The watch-and-process loop
    server_task = asyncio.create_task(receiver.start_server())
    process_task = asyncio.create_task(watch_and_process(chatbot))

    # Wait for both tasks (the server never ends, nor does the watcher)
    await asyncio.gather(server_task, process_task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server shutdown via KeyboardInterrupt")
