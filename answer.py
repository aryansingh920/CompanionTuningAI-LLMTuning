"""
Created on 16/03/2025

@author: Aryan

Filename: answer.py

Relative Path: answer.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
import os
from pydub import AudioSegment
from pydub.playback import play
import re


def load_model(model_path):
    """Load the fine-tuned model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Check for available device
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Move model to appropriate device
    model = model.to(device)
    return model, tokenizer, device


def generate_response(prompt, model, tokenizer, device, max_length=100):
    """Generate a response from the model given a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt",
                       padding=True, truncation=True).to(device)

    # Ensure attention_mask is provided explicitly
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length + input_ids.shape[1],
        temperature=0.6,  # Lower temperature for more focused responses
        top_p=0.92,
        repetition_penalty=1.2,  # Add repetition penalty to prevent loops
        no_repeat_ngram_size=3,  # Prevent repeating of 3-word phrases
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    response = response[len(prompt):]  # Remove prompt from response
    return clean_response(response.strip())


def clean_response(response):
    """Clean and filter the model's response to ensure it's appropriate."""
    # Remove repetitive phrases
    if "answerText:" in response:
        response = response.split("answerText:")[0].strip()

    # Handle repetitive text patterns
    response = re.sub(r'(.{20,}?)\1+', r'\1', response)

    # Check for inappropriate content or patterns
    inappropriate_phrases = [
        "cheating",
        "Not cheating answerText:",
        "can't know he",
        "If he is not"
    ]

    for phrase in inappropriate_phrases:
        if phrase in response:
            return "I understand you're going through a difficult time. Could you tell me more about what's troubling you?"

    # Check if the response is too short or empty
    if len(response.strip()) < 15:
        return "I'm here to help. Could you share more about what you're experiencing?"

    return response


def text_to_speech(text):
    """Convert text to speech and play it."""
    if not text.strip():
        return  # Skip if response is empty

    temp_audio_file = "response.mp3"

    try:
        tts = gTTS(text=text, lang="en", slow=False)
        tts.save(temp_audio_file)

        # Load and play the audio using pydub
        audio = AudioSegment.from_mp3(temp_audio_file)
        play(audio)

    except Exception as e:
        print(f"Error in text-to-speech: {e}")

    finally:
        if os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)  # Cleanup


def chat_interface():
    """Simple command-line chat interface with speech output."""
    print("Loading the therapy model... (this may take a moment)")
    model_path = "./fine_tuned_therapy_gpt"
    model, tokenizer, device = load_model(model_path)

    print("\n===== Therapy Chatbot =====")
    print("Type your questions or concerns below. Type 'exit' to quit.")

    # Define fallback responses for when the model completely fails
    fallback_responses = [
        "I'm listening. Can you tell me more about that?",
        "That sounds challenging. How has this been affecting you?",
        "I'm here to support you. What would be most helpful to discuss right now?",
        "Thank you for sharing. How long have you been feeling this way?",
        "I understand this is important to you. Could we explore that further?"
    ]
    fallback_index = 0

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        # Formatting the prompt with guidance for the model
        prompt = f"As a professional, empathetic therapist, provide helpful and ethical advice. Focus on understanding the person's concerns without making assumptions.\nquestionText: {user_input} answerText: "

        print("\nThinking...")
        try:
            response = generate_response(prompt, model, tokenizer, device)

            # Check if response seems problematic
            problematic_patterns = ["answerText:",
                                    "questionText:", "thinking...", "prompt:"]
            if any(pattern in response.lower() for pattern in problematic_patterns) or len(response) < 10:
                # Use fallback response
                response = fallback_responses[fallback_index]
                fallback_index = (fallback_index + 1) % len(fallback_responses)

            print(f"Therapist: {response.strip()}")
            text_to_speech(response)

        except Exception as e:
            print(f"Error generating response: {e}")
            # Use fallback response in case of errors
            fallback = fallback_responses[fallback_index]
            fallback_index = (fallback_index + 1) % len(fallback_responses)
            print(f"Therapist: {fallback}")
            text_to_speech(fallback)


if __name__ == "__main__":
    chat_interface()
