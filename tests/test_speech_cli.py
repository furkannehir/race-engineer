import asyncio
from pathlib import Path

import pytest
from test_stt_audio import tone

from race_engineer.application.speech_cli import spoken_turn
from race_engineer.core.speech_input import Transcription


class Recognizer:
    def __init__(self, result):
        self.result = result

    async def transcribe(self, audio):
        return self.result


class Conversation:
    def __init__(self):
        self.questions = []

    async def ask(self, question, **kwargs):
        self.questions.append((question, kwargs))
        return "reply"


def test_only_successful_transcripts_reach_conversation():
    for status in ("no_speech", "unsupported_language"):
        recognizer = Recognizer(Transcription(status=status, audio_duration_s=0.4))
        conversation = Conversation()
        _, reply = asyncio.run(spoken_turn(recognizer, conversation, tone()))
        assert reply is None
        assert not conversation.questions


def test_turkish_transcript_and_language_override():
    recognizer = Recognizer(
        Transcription(status="transcribed", text="Kaçıncıyız?", language="tr", audio_duration_s=0.4)
    )
    conversation = Conversation()
    asyncio.run(spoken_turn(recognizer, conversation, tone()))
    assert conversation.questions == [("Kaçıncıyız?", {"reply_language": "tr"})]
    asyncio.run(spoken_turn(recognizer, conversation, tone(), reply_language="en"))
    assert conversation.questions[-1][1] == {"reply_language": "en"}


def test_voice_file_path_does_not_open_microphone(monkeypatch, capsys):
    from race_engineer.application.speech_cli import voice_replay
    from race_engineer.core.conversation import ConversationPlan

    root = Path(__file__).parents[1]
    closed = []

    class FakeRecognizer(Recognizer):
        def __init__(self, config):
            super().__init__(
                Transcription(
                    status="transcribed", text="Where are we?", language="en", audio_duration_s=0.4
                )
            )

        async def aclose(self):
            closed.append(True)

    class Planner:
        def __init__(self, config):
            pass

        async def plan(self, request):
            return ConversationPlan(language="en", queries=("position",), clarification="none")

    def microphone(*args, **kwargs):
        raise AssertionError("file mode must not open a microphone")

    monkeypatch.setattr("race_engineer.application.speech_cli.QwenSpeechRecognizer", FakeRecognizer)
    monkeypatch.setattr("race_engineer.application.speech_cli.conversation_planner", Planner)
    monkeypatch.setattr("race_engineer.application.speech_cli.PushToTalkMicrophone", microphone)
    monkeypatch.setattr("race_engineer.application.speech_cli.load_wav", lambda *a, **kw: tone())
    code = asyncio.run(
        voice_replay(
            root / "config/default.toml",
            root / "fixtures/synthetic/conversation",
            audio_path=Path("test.wav"),
            text_only=True,
        )
    )
    assert code == 0 and closed == [True]
    assert "P6" in capsys.readouterr().out


@pytest.mark.parametrize("language", ["en", "tr"])
@pytest.mark.parametrize("failure", [None, "start", "speak"])
def test_interactive_audio_handoff_and_text_fallback(monkeypatch, capsys, language, failure):
    from race_engineer.application.speech_cli import voice_replay
    from race_engineer.core.conversation import ConversationPlan
    from race_engineer.core.speech_output import SpeechOutputError, SpeechOutputResult

    root = Path(__file__).parents[1]
    steps = []
    captured = []

    class FakeRecognizer(Recognizer):
        def __init__(self, config):
            super().__init__(
                Transcription(
                    status="transcribed",
                    text="Where are we?",
                    language=language,
                    audio_duration_s=0.4,
                )
            )

        async def start(self):
            steps.append("asr_start")

        async def aclose(self):
            steps.append("asr_close")

    class Planner:
        def __init__(self, config):
            pass

        async def plan(self, request):
            steps.append("plan")
            return ConversationPlan(language=language, queries=("position",), clarification="none")

    class Microphone:
        def __init__(self, config):
            self.turns = iter((tone(), None))

        async def next_clip(self):
            steps.append("capture")
            return next(self.turns)

        async def aclose(self):
            steps.append("microphone_close")

    class Speaker:
        def __init__(self, config):
            assert config.output_device == 7

        async def start(self):
            steps.append("tts_start")
            if failure == "start":
                raise SpeechOutputError("tts_runtime_missing")

        async def speak(self, reply):
            steps.append("speak")
            captured.append(reply)
            if failure == "speak":
                raise SpeechOutputError("tts_playback_timeout")
            await asyncio.sleep(0)
            # A second microphone capture must not start during the spoken answer.
            assert steps.count("capture") == 1
            steps.append("playback_finished")
            return SpeechOutputResult(
                language=reply.language,
                played=True,
                audio_duration_s=0.5,
                synthesis_ms=10,
                playback_ms=500,
            )

        async def aclose(self):
            steps.append("tts_close")

    monkeypatch.setattr("race_engineer.application.speech_cli.QwenSpeechRecognizer", FakeRecognizer)
    monkeypatch.setattr("race_engineer.application.speech_cli.conversation_planner", Planner)
    monkeypatch.setattr("race_engineer.application.speech_cli.PushToTalkMicrophone", Microphone)
    monkeypatch.setattr("race_engineer.application.speech_cli.PiperConversationSpeaker", Speaker)
    code = asyncio.run(
        voice_replay(
            root / "config/default.toml", root / "fixtures/synthetic/conversation", output_device=7
        )
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "Engineer:" in output
    assert steps.count("capture") == 2
    assert steps.count("tts_close") == 1
    assert "microphone_close" in steps and "asr_close" in steps
    if failure == "start":
        assert "Continuing with text replies" in output
        assert not captured
    else:
        assert len(captured) == 1 and captured[0].language == language
        assert captured[0].text in output
        if failure == "speak":
            assert "The text reply is above" in output
        else:
            assert steps.index("playback_finished") < len(steps) - 4


def test_transcript_displayed_before_conversation_inference():
    seen = []
    conversation = Conversation()

    def display(transcript):
        assert not conversation.questions
        seen.append(transcript.text)

    recognizer = Recognizer(
        Transcription(status="transcribed", text="Position?", language="en", audio_duration_s=0.4)
    )
    asyncio.run(spoken_turn(recognizer, conversation, tone(), on_transcript=display))
    assert seen == ["Position?"]
