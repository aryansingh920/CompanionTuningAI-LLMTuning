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
import time


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

        # Queues for audio processing
        self.audio_queue = queue.Queue()
        self.response_queue = queue.Queue()

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
            self.audio_queue.put(indata.copy())

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='float32',
            callback=callback,
            blocksize=int(self.sample_rate * self.block_duration)
        ):
            sd.sleep(999999)  # Keep thread alive

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
                max_length=200,
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

    def process_thread(self):
        """Process audio and generate responses."""
        accumulated_audio = []

        while True:
            try:
                # Accumulate audio blocks
                while len(accumulated_audio) < 6:  # Collect ~3 seconds
                    audio_block = self.audio_queue.get(timeout=1)
                    accumulated_audio.append(audio_block)

                # Combine audio blocks
                full_audio = np.concatenate(accumulated_audio)

                # Transcribe
                transcript = self.transcribe_audio(full_audio)
                print(f"Transcribed: {transcript}")

                if transcript.strip().lower() in ['exit', 'quit', 'q']:
                    break

                # Generate response
                prompt = f"As a professional, empathetic therapist, provide helpful and ethical advice. Focus on understanding the person's concerns without making assumptions.\nquestionText: {transcript} answerText: "
                response = self.generate_response(prompt)

                print(f"Response: {response}")

                # Text-to-speech
                self.text_to_speech(response)

                # Reset accumulated audio
                accumulated_audio = []

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                accumulated_audio = []

    def start(self):
        """Start the therapy chatbot."""
        print("\n===== Low-Latency Therapy Chatbot =====")
        print("Speak naturally. Say 'exit' to quit.")

        # Start audio capture thread
        capture_thread = threading.Thread(
            target=self.audio_capture_thread, daemon=True)
        capture_thread.start()

        # Start processing thread
        process_thread = threading.Thread(
            target=self.process_thread, daemon=True)
        process_thread.start()

        # Keep main thread alive
        process_thread.join()


def main():
    chatbot = TherapyChatbot()
    chatbot.start()


if __name__ == "__main__":
    main()
