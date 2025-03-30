import os
import asyncio
from TherapyChatbot import TherapyChatbot
from WebSocketAudioReceiver import WebSocketAudioReceiver


async def watch_and_process(chatbot: TherapyChatbot, receiver: WebSocketAudioReceiver):
    """
    Continuously monitors 'input' folder for .wav files (1.wav, 2.wav, etc.),
    transcribes, calls GPT, does TTS, writes output/<same_number>.mp3, then
    sends 'RESPONSE_READY_ABS /abs/path/to/the/file.mp3' to the client.
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

            processed_files.add(wav_file)

            # 4) Remove the input WAV
            try:
                os.remove(input_path)
                print(f"Removed input file: {input_path}")
            except Exception as e:
                print(f"Error removing {input_path}: {e}")

            # 5) Broadcast the absolute path of the MP3 to Unity
            abs_mp3_path = os.path.abspath(output_path)
            msg = f"RESPONSE_READY_ABS {abs_mp3_path}"
            print(f"Broadcasting: {msg}")
            await receiver.broadcast_message(msg)

        await asyncio.sleep(2)


async def main():
    chatbot = TherapyChatbot()
    receiver = WebSocketAudioReceiver(host='localhost', port=8080)

    server_task = asyncio.create_task(receiver.start_server())
    process_task = asyncio.create_task(watch_and_process(chatbot, receiver))

    await asyncio.gather(server_task, process_task)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server shutdown via KeyboardInterrupt")
