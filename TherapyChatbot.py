import re
import torch
import whisper
from gtts import gTTS
from transformers import AutoModelForCausalLM, AutoTokenizer



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
