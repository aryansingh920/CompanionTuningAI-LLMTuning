import os
import asyncio
from TherapyChatbot import TherapyChatbot
from WebSocketAudioReceiver import WebSocketAudioReceiver



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
