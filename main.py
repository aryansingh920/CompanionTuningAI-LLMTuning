import os
import asyncio
from TherapyChatbot import TherapyChatbot
from WebSocketAudioReceiver import WebSocketAudioReceiver


async def watch_and_process(chatbot: TherapyChatbot, receiver: WebSocketAudioReceiver):
    """
    Continuously monitors 'input' folder for .wav files (1.wav, 2.wav, etc.),
    transcribes, calls GPT, does TTS, then streams the MP3 audio bytes directly
    to the connected clients.
    """
    os.makedirs('output', exist_ok=True)
    processed_files = set()

    while True:
        all_wav = [f for f in os.listdir('input') if f.endswith('.wav')]
        unprocessed = [f for f in all_wav if f not in processed_files]

        def numeric_key(filename):
            try:
                return int(os.path.splitext(filename)[0])
            except ValueError:
                return 999999

        unprocessed.sort(key=numeric_key)

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

            # 4) Read the MP3 file and send its bytes directly
            try:
                with open(output_path, 'rb') as mp3_file:
                    audio_bytes = mp3_file.read()
                    # Send audio data directly through WebSocket
                    await receiver.send_audio_data(audio_bytes)
                    print(f"Streamed {len(audio_bytes)} bytes of audio data")
                    # remove the MP3 file after sending
                    os.remove(output_path)
                    print(f"Removed MP3 file: {output_path}")
            except FileNotFoundError:
                print(f"MP3 file not found: {output_path}")
            except PermissionError:
                print(f"Permission denied for MP3 file: {output_path}")
            except Exception as e:
                print(f"Error reading/sending MP3 file: {e}")

            processed_files.add(wav_file)

            # 5) Remove the input WAV
            try:
                os.remove(input_path)
                print(f"Removed input file: {input_path}")
            except Exception as e:
                print(f"Error removing {input_path}: {e}")

        await asyncio.sleep(2)


async def main():
    chatbot = TherapyChatbot(model_path="fine_tuned_tone_adaptive_model")
    receiver = WebSocketAudioReceiver(host='0.0.0.0', port=8080)

    server_task = asyncio.create_task(receiver.start_server())
    process_task = asyncio.create_task(watch_and_process(chatbot, receiver))

    await asyncio.gather(server_task, process_task)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server shutdown via KeyboardInterrupt")
