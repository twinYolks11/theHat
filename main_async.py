import asyncio
import time
import os

from theHat.input_handlers import get_user_query
from theHat.agents import OpenWebUIAgent
from theHat.tts_models import OpenAISTTModel
from theHat.audio_utils import play_audio

from google import genai
from google.generativeai.types import GenerateContentConfig

async def main():
    # Get user query
    t = time.time()
    query = await get_user_query()
    print(f"Received input: {query} in {time.time() - t:.2f} sec")

    # Get agent response
    t = time.time()

    #configure the content generation to request audio output
    config = GenerateContentConfig(
        response_modalities=['AUDIO']
    )

    # agent = OpenWebUIAgent()
    # response = agent.get_response(query)
    # print(f"Response: {response}")
    # print(f"Time taken = {time.time() - t:.2f} sec")
    client = genai.Client()
    response = client.models.generate_content(model="gemini-2.5-flash-tts", contents=query, generation_config=config)
    #print(response)

    # Convert response to audio
    t = time.time()
    # output_path = f"agent_reponse.mp3"
    # model_tts = OpenAISTTModel()
    # model_tts.generate_audio(response, output_path)

    if response.audio:
        audio_file_path="output_audio.mp3"
        with open(audio_file_path, "wb") as f:
            f.write(response.audio.data)
            print(f"Audio saved to {audio_file_path}")
    else:
        print("No audio data received in the response")
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