import torch
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
import os
import re
import queue
import threading


class TherapyChatbot:
    def __init__(self, model_path="./fine_tuned_therapy_gpt"):
        # Speech Recognition Setup
        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model("base")

        # Therapy Model Setup
        print("Loading therapy model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)

        # Device Selection
        self.device = torch.device("mps" if torch.backends.mps.is_available() else
                                   ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Using device: {self.device}")
        self.model = self.model.to(self.device)

        # Audio Configuration
        self.sample_rate = 16000  # Whisper's preferred sample rate
        self.block_duration = 0.5  # 500ms blocks for low latency
        self.channels = 1

        # Audio accumulation
        self.audio_queue = queue.Queue()
        self.accumulated_audio = []
        self.is_listening = False
        self.transcript_ready = threading.Event()

        # Fallback Responses
        self.fallback_responses = [
            "I'm listening. Can you tell me more about that?",
            "That sounds challenging. How has this been affecting you?",
            "I'm here to support you. What would be most helpful to discuss right now?",
            "Thank you for sharing. How long have you been feeling this way?",
            "I understand this is important to you. Could we explore that further?"
        ]
        self.fallback_index = 0

    def audio_capture_thread(self):
        """Continuously capture audio in the background."""
        def callback(indata, frames, time, status):
            if status:
                print(status)
            if self.is_listening:
                self.audio_queue.put(indata.copy())

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='float32',
            callback=callback,
            blocksize=int(self.sample_rate * self.block_duration)
        ):
            while True:
                sd.sleep(100)

    def listening_thread(self):
        """Manage listening state and audio accumulation."""
        while True:
            # Start listening
            self.is_listening = True
            self.accumulated_audio = []
            print("\nListening... (Press Enter when done speaking)")

            # Accumulate audio while listening
            while self.is_listening:
                try:
                    audio_block = self.audio_queue.get(timeout=0.1)
                    self.accumulated_audio.append(audio_block)
                except queue.Empty:
                    continue

    def input_thread(self):
        """Listen for Enter key to stop listening and process transcript."""
        while True:
            input()  # Wait for Enter key
            if self.is_listening:
                self.is_listening = False
                self.transcript_ready.set()

    def transcribe_audio(self, audio_data):
        """Transcribe audio using Whisper."""
        # Temporary file for transcription
        sf.write('temp_input.wav', audio_data, self.sample_rate)
        result = self.whisper_model.transcribe('temp_input.wav')
        os.remove('temp_input.wav')
        return result['text']

    def generate_response(self, prompt):
        """Generate response from therapy model."""
        try:
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
            response = response[len(prompt):].strip()

            return self.clean_response(response)
        except Exception as e:
            print(f"Response generation error: {e}")
            return self.get_fallback_response()

    def clean_response(self, response):
        """Clean and filter the model's response."""
        # Remove repetitive phrases and patterns
        response = re.sub(r'(.{20,}?)\1+', r'\1', response)

        inappropriate_phrases = [
            "answerText:", "cheating", "questionText:"
        ]

        for phrase in inappropriate_phrases:
            if phrase in response:
                return self.get_fallback_response()

        return response if len(response) > 10 else self.get_fallback_response()

    def get_fallback_response(self):
        """Get a fallback response and cycle through them."""
        response = self.fallback_responses[self.fallback_index]
        self.fallback_index = (self.fallback_index +
                               1) % len(self.fallback_responses)
        return response

    def text_to_speech(self, text):
        """Convert text to speech with low latency."""
        try:
            tts = gTTS(text=text, lang="en", slow=False)
            tts.save("response.mp3")

            # Play audio using sounddevice for minimal latency
            data, fs = sf.read("response.mp3")
            sd.play(data, fs)
            sd.wait()

            os.remove("response.mp3")
        except Exception as e:
            print(f"Text-to-speech error: {e}")

    def process_transcript_thread(self):
        """Process transcripts when ready."""
        while True:
            # Wait for transcript to be ready
            self.transcript_ready.wait()
            self.transcript_ready.clear()

            # Check if we have accumulated audio
            if self.accumulated_audio:
                # Combine audio blocks
                full_audio = np.concatenate(self.accumulated_audio)

                # Transcribe
                transcript = self.transcribe_audio(full_audio)
                print(f"\nTranscribed: {transcript}")

                # Generate response
                prompt = f"As a professional, empathetic therapist, provide helpful and ethical advice. Focus on understanding the person's concerns without making assumptions.\nquestionText: {transcript} answerText: "
                response = self.generate_response(prompt)

                print(f"\nResponse: {response}")

                # Text-to-speech
                self.text_to_speech(response)

    def start(self):
        """Start the therapy chatbot."""
        print("\n===== Low-Latency Therapy Chatbot =====")
        print("Press Enter to start/stop listening.")

        # Start audio capture thread
        capture_thread = threading.Thread(
            target=self.audio_capture_thread, daemon=True)
        capture_thread.start()

        # Start listening thread
        listening_thread = threading.Thread(
            target=self.listening_thread, daemon=True)
        listening_thread.start()

        # Start input thread
        input_thread = threading.Thread(
            target=self.input_thread, daemon=True)
        input_thread.start()

        # Start transcript processing thread
        process_thread = threading.Thread(
            target=self.process_transcript_thread, daemon=True)
        process_thread.start()

        # Keep main thread alive
        input_thread.join()


def main():
    chatbot = TherapyChatbot()
    chatbot.start()


if __name__ == "__main__":
    main()
