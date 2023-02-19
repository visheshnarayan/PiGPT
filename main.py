# imports 
# pip3 install -q git+https://github.com/openai/whisper.git
# pip3 install gradio
import openai
import gradio as gr
import whisper
import warnings
import os

# don't want warnings popping up
warnings.filterwarnings("ignore")

# initialize variables
engine_type='text-davinci-003'
openai.api_key=os.getenv("gpt_key")
max_tokens=1024

class PiGPT:
    '''A virtual assistant using a speech to text to OpenAI pipeline'''

    def __init__(self, engine, max_tokens):

        '''Initialization function'''

        self.engine=engine
        self.max_tokens=max_tokens
        self.model=whisper.load_model("base")

    def transcribe(self, audio):

        '''Records and transcribes audio and gets response from GPT model'''

        # load audio
        audio=whisper.load_audio(audio)
        audio=whisper.pad_or_trim(audio)

        # create spectrogram and send to processing device
        spec=whisper.log_mel_spectrogram(audio).to(self.model.device)

        # detect language
        _, probs=self.model.detect_language(spec)

        # decode
        options=whisper.DecodingOptions(fp16 = False)
        result=whisper.decode(self.model, spec, options)
        result_text=result.text

        # pass transcribed text to ChatGPT
        # * request
        response = openai.Completion.create(
            engine=engine_type,
            # prompt="what is your name",
            prompt=result_text,
            max_tokens=max_tokens,
            temperature=0.5,
            top_p=1
        )
        # return transcribed text and ChatGPT response
        return [result_text, response.choices[0].text]

    def run(self):

        '''Launches web GUI for Gradio'''
        
        # create text boxes for UI
        out1=gr.Textbox(label="Speech to Text")
        out2=gr.Textbox(label="ChatGPT")

        gr.Interface(
            title="PiGPT UI",
            fn=self.transcribe,
            inputs=[gr.inputs.Audio(source='microphone', type='filepath')],
            outputs=[out1, out2],
            live=True
        ).launch()

if __name__ == "__main__":
    PiGPT(engine=engine_type, max_tokens=max_tokens).run()