import asyncio
import time
import os
import wave

from theHat.input_handlers import get_user_query
from theHat.agents import OpenWebUIAgent
from theHat.tts_models import OpenAISTTModel
from theHat.audio_utils import play_audio

from google import genai
from google.genai import types

# Set up the wave file to save the output:
def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
   with wave.open(filename, "wb") as wf:
      wf.setnchannels(channels)
      wf.setsampwidth(sample_width)
      wf.setframerate(rate)
      wf.writeframes(pcm)

async def main():
    # Get user query
    t = time.time()
    query = await get_user_query()
    print(f"Received input: {query} in {time.time() - t:.2f} sec")

    # Get agent response
    t = time.time()

    client = genai.Client()
    transcript = client.models.generate_content(
        model="gemini-2.5-flash",
        contents='respond to the following question with a concise answer of 100 words or less' \
        ' as if it was answered by the sorting hat from hogwarts: ' + query
    ).text

    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts", 
        contents=transcript,
        config=types.GenerateContentConfig(
            response_modalities=['AUDIO'],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name='Sadaltager',
                    )
                )
            )
        )
    )

    # Convert response to audio
    t = time.time()

    audio_out = response.candidates[0].content.parts[0].inline_data.data
    audio_file_path='response.wav'
    wave_file(audio_file_path, audio_out)
    print(f"Time taken = {time.time() - t:.2f} sec")

    # Play agent response
    t = time.time()
    play_audio(audio_file_path)
    print(f"Played audio in {time.time() - t:.2f} sec")

    # cleanup
    os.remove(audio_file_path)


if __name__ == "__main__":
    while True:
        print("Waiting for input ...")
        asyncio.run(main())    