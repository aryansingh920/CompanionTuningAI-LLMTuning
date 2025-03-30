import asyncio
import numpy as np
import soundfile as sf
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole


class AudioStreamTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.audio_frames = []

    async def recv(self):
        frame = await self.track.recv()
        self.audio_frames.append(frame)
        return frame

    def save_audio(self):
        if self.audio_frames:
            print(f"Saving {len(self.audio_frames)} audio frames...")
            audio_data = b''.join([f.planes[0].to_bytes()
                                  for f in self.audio_frames])
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            sf.write('output.wav', audio_array, 48000)
            print("Saved to output.wav")


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])

    pc = RTCPeerConnection()
    recorder = MediaBlackhole()
    audio_track_ref = {'track': None}

    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            print("Audio track received")
            audio_stream = AudioStreamTrack(track)
            audio_track_ref['track'] = audio_stream
            asyncio.ensure_future(recorder.start())

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    async def stop():
        await pc.close()
        if audio_track_ref['track']:
            audio_track_ref['track'].save_audio()

    request.app.on_shutdown.append(lambda app: stop())
    return web.json_response({'sdp': pc.localDescription.sdp, 'type': pc.localDescription.type})

app = web.Application()
app.router.add_post('/offer', offer)

web.run_app(app, port=8080)
