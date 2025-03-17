import asyncio
import signal
from dotenv import load_dotenv

from pydantic_settings import BaseSettings, SettingsConfigDict
from vocode.helpers import create_streaming_microphone_input_and_speaker_output
from vocode.logging import configure_pretty_logging
from vocode.streaming.agent.langchain_agent import LangchainAgent
from vocode.streaming.models.agent import LangchainAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import RimeSynthesizerConfig
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
)
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.synthesizer.rime_synthesizer import RimeSynthesizer
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber

from rag import build_vectorstore, setup_chain

load_dotenv()
configure_pretty_logging()


async def main():
    vectorstore = build_vectorstore("nutrition.pdf")
    rag_chain = setup_chain(vectorstore)

    # create microphone input and speaker output
    (microphone_input, speaker_output) = create_streaming_microphone_input_and_speaker_output(
        use_default_devices=True
    )
    
    agent = LangchainAgent(
        agent_config=LangchainAgentConfig(
            prompt_preamble="You are an expert nutritionist. Answer the question succinctly and as accurately as possible using the provided context,"
            "and do not use any symbols like * in your answer.",
            initial_message=BaseMessage(text="Hello! How can I help you today?"),
            model_name="gpt-4o-mini",
            provider="openai",
            temperature=0.0,
            max_tokens=150,
        ),
        chain=rag_chain,
    )

    conversation = StreamingConversation(
        output_device=speaker_output,
        transcriber=DeepgramTranscriber(
            DeepgramTranscriberConfig.from_input_device(
                microphone_input,
                endpointing_config=PunctuationEndpointingConfig(),
            ),
        ),
        agent=agent, # pass our langchain agent
        synthesizer=RimeSynthesizer(
            RimeSynthesizerConfig.from_output_device(
                speaker_output,
            )
        )
    )

    await conversation.start()
    print("Conversation started, press Ctrl+C to end")
    signal.signal(signal.SIGINT, lambda _0, _1: asyncio.create_task(conversation.terminate()))

    while conversation.is_active():
        chunk = await microphone_input.get_audio()
        conversation.receive_audio(chunk)


if __name__ == "__main__":
    asyncio.run(main())