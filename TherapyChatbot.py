import re
import torch
import whisper
from gtts import gTTS
from transformers import AutoModelForCausalLM, AutoTokenizer


class TherapyChatbot:
    """Tone‑adaptive chatbot focused on supportive conversation without claiming professional status."""

    def __init__(self, model_path: str = "./tone_adaptive_chatbot"):
        # 1. Speech‑to‑text
        print("Loading Whisper model…")
        self.whisper_model = whisper.load_model(
            "base")  # choose size as needed

        # 2. Language model
        print("Loading therapy model…")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)

        # 3. Device
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else (
                "cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Using device: {self.device}")
        self.model = self.model.to(self.device)

        # 4. Conversation memory
        self.conversation_history: list[str] = []
        self.max_history_tokens: int = 512

        # 5. Fallback responses (rotating)
        self.fallback_responses: list[str] = [
            "I'm here to listen. Would you like to share more about what's on your mind?",
            "That sounds challenging. How has this situation been affecting you?",
            "I'm here to support you. What would be most helpful to discuss right now?",
            "Thank you for sharing. How long have you been experiencing this?",
            "I understand this is important to you. Could we explore that further?",
            "It sounds like you're dealing with a lot. What aspect is most concerning for you?",
            "I appreciate you opening up. What kinds of things have you tried so far?",
            "I'm listening. How does this situation make you feel?",
            "That's certainly worth discussing. Could you tell me more about what happened?",
            "I'm here to help. What would be a good first step for us to explore together?",
        ]
        self.fallback_index: int = 0

    # ---------------------------------------------------------------------
    # Audio → text
    # ---------------------------------------------------------------------
    def transcribe_audio(self, file_path: str) -> str:
        """Transcribe a WAV/MP3 file using Whisper."""
        print(f"Transcribing {file_path}…")
        result = self.whisper_model.transcribe(file_path)
        return result["text"].strip()

    # ---------------------------------------------------------------------
    # Text → response
    # ---------------------------------------------------------------------
    def generate_response(self, transcript: str) -> str:
        """Generate an empathetic response with multi‑candidate selection and safety filters."""
        # 1. Add user input to history
        self.conversation_history.append(f"Person: {transcript}")

        # 2. Build context within token window
        context, history_tokens = "", 0
        for msg in reversed(self.conversation_history[:-1]):
            tokens = len(self.tokenizer.encode(msg))
            if history_tokens + tokens < self.max_history_tokens:
                context = msg + "\n" + context
                history_tokens += tokens
            else:
                break

        prompt = (
            "You are an AI assistant providing supportive responses. Respond with empathy "
            "and understanding, but never claim to be a human professional or expert.\n\n" +
            context +
            f"Person: {transcript}\nAI Assistant: "
        )

        try:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)

            num_candidates = 3
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=150,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                num_return_sequences=num_candidates,
                do_sample=True,
            )

            # --- Fix: slice off prompt tokens before decoding ---
            input_ids = inputs["input_ids"]
            prompt_len = input_ids.shape[1]

            cleaned_candidates: list[str] = []
            for out in outputs:
                gen_ids = out[prompt_len:]
                resp = self.tokenizer.decode(
                    gen_ids, skip_special_tokens=True).strip()
                resp = resp.split("\n")[0].strip()
                cleaned = self._clean_response(resp)
                if cleaned:
                    cleaned_candidates.append(cleaned)

            if not cleaned_candidates:
                return self._get_fallback_response()

            best = self._select_best_response(cleaned_candidates, transcript)
            final = self._post_process_response(best, transcript)

            # Save assistant reply to history
            self.conversation_history.append(f"AI Assistant: {final}")
            self.conversation_history = self.conversation_history[-20:]

            return final

        except Exception as exc:
            print(f"Error generating response: {exc}")
            return self._get_fallback_response()

    # ------------------------------------------------------------------
    # Candidate scoring & filtering helpers
    # ------------------------------------------------------------------
    def _select_best_response(self, candidates: list[str], transcript: str) -> str:
        """Score candidates and return the best."""
        identity_patterns = re.compile(
            # false‑identity
            r"i am (?:a|an)? (therapist|doctor|professional|counselor)|"
            r"my name is|"  # personal disclosure
            r"i(?:'m| am) certified|"
            r"i've been (?:working|practicing)",
            re.IGNORECASE,
        )
        empathy_keywords = ["understand", "hear you",
                            "that sounds", "feel", "difficult", "challenging"]

        best_score, best_resp = -float("inf"), None
        for resp in candidates:
            score = 0

            if identity_patterns.search(resp):
                score -= 100
            if any(tag in resp.lower() for tag in ["question:", "answer:", "answertext:", "questiontext:"]):
                score -= 50
            non_ascii_ratio = sum(ord(c) > 127 for c in resp) / len(resp)
            if non_ascii_ratio > 0.2:
                score -= 75
            score += sum(2 for kw in empathy_keywords if kw in resp.lower())
            if 30 <= len(resp) <= 150:
                score += 10

            if score > best_score:
                best_score, best_resp = score, resp

        return best_resp if best_score >= -20 else self._get_fallback_response()

    def _clean_response(self, response: str) -> str:
        """Remove artefacts, repeated patterns, and unsafe content."""
        if not response:
            return ""

        # Deduplicate long repeated substrings
        response = re.sub(r"(.{20,}?)\1+", r"\1",
                          response, flags=re.IGNORECASE)

        # Strip leftover markers
        response = re.sub(r"(answer|question)(Text|:)", "",
                          response, flags=re.IGNORECASE).strip()

        # Capitalize
        if response:
            response = response[0].upper() + response[1:]

        # Block problematic patterns
        bad_patterns = [
            r"i am (?:a|an)? (therapist|counselor|professional|doctor)",
            r"my name is",
            r"i have been (?:a|an|the) (therapist|counselor)",
            r"my therapist", r"my counselor",
            r"questiontext", r"answertext", r"answer:", r"question:",
        ]
        if any(re.search(pat, response, re.IGNORECASE) for pat in bad_patterns):
            return ""

        if sum(ord(c) > 127 for c in response) / len(response) > 0.15:
            return ""

        if not 15 <= len(response) <= 200:
            return ""

        return response

    def _post_process_response(self, response: str, transcript: str) -> str:
        """Final sanity check before output."""
        if not transcript.strip() and len(response) > 40:
            return self._get_fallback_response()

        crisis_triggers = ["suicide", "kill",
                           "hurt myself", "end my life", "self-harm"]
        if any(word in transcript.lower() for word in crisis_triggers):
            return (
                "I notice you mentioned something concerning. If you're in crisis, please "
                "contact a crisis helpline like 988 (US) or reach out to a mental health professional immediately."
            )

        if response.count(".") > 10:  # trim rambling
            response = ". ".join(response.split(".")[:3]).strip()
            if not response.endswith("."):
                response += "."

        return response

    # ------------------------------------------------------------------
    # Fallback + TTS utilities
    # ------------------------------------------------------------------
    def _get_fallback_response(self) -> str:
        resp = self.fallback_responses[self.fallback_index]
        self.fallback_index = (self.fallback_index +
                               1) % len(self.fallback_responses)
        return resp

    def text_to_speech(self, text: str, output_file: str):
        """Convert text to speech and save as MP3 using gTTS."""
        try:
            gTTS(text=text, lang="en", slow=False).save(output_file)
            print(f"TTS saved to {output_file}")
        except Exception as exc:
            print(f"Text‑to‑speech error: {exc}")
