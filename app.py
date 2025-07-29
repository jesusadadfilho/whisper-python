import gradio as gr
import whisper

model = whisper.load_model("tiny")

def transcrever(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]

iface = gr.Interface(
    fn=transcrever,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    title="Whisper STT API"
)

iface.launch(show_error=True)
