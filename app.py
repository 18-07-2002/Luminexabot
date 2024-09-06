import os
import gradio as gr
import logging
from textwrap import dedent
import json
import requests
from gradio import Image
import numpy as np
from pyChatGPT import ChatGPT
import whisper
import io
from PIL import Image
import numpy as np
from PIL import Image
from diffusers import DiffusionPipeline
import torch
from typing import Dict, Any
from huggingface_hub import login
import os
import modin.pandas as pd
from diffusers.models import AutoencoderKL
from transformers import AutoTokenizer
import speech_recognition as sr
import ffmpeg
import whisper
from pytube import YouTube
import re
import warnings
from IPython.display import display
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import functools
import torch
from fabric.generator import AttentionBasedGenerator
import openai
import config
import subprocess

login(token="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")
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logging.warning("READY. App started...")


# TEXT TO IMAGE
# METHOD-1
# def generate_image(prompt, api_key):
#     # Set up the request headers
#     headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

#     # Set up the request data
#     payload = {
#         "model": "stable-Diffusion-v1-5",
#         "text_prompts": prompt,
#         "cfg_scale": 7,
#         "clip_guidance_preset": "FAST_BLUE",
#         "height": 512,
#         "width": 512,
#         "samples": 1,
#         "steps": 30,
#     }

#     # Send the request to Dream's API
#     response = requests.post(
#         "https://api.stability.ai/v1/generation/stable-diffusion-v1-5/text-to-image",
#         json=payload,
#         headers=headers,
#         stream=True,
#     )
#     response.raise_for_status()

#     # Extract the image URL from the response
#     image_url = response.json()["payload"][0]["url"]

#     # Download and display the image using PIL and Gradio
#     image_bytes = requests.get(image_url).content
#     image = Image.open(io.BytesIO(image_bytes))
#     image_arr = np.array(image)
#     img = image_arr[0]
#     img.save("sd_image.png")
#     return img


# Method-2
# model_name = "dreamlike-art/dreamlike-photoreal-2.0"
# model_name = ""
# model_ckpt = "https://huggingface.co/Lykon/DreamShaper/blob/main/DreamShaper_7_pruned.safetensors"


# class GeneratorWrapper:
#     def __init__(self, model_name=None, model_ckpt=None):
#         self.model_name = model_name if model_name else None
#         self.model_ckpt = model_ckpt if model_ckpt else None
#         self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"

#         self.reload()

#     def generate(self, *args, **kwargs):
#         if not hasattr(self, "generator"):
#             self.reload()
#         print(f"Device for model and tensors: {self.device}")
#         return self.generator.generate(*args, **kwargs)

#     def to(self, device):
#         return self.generator.to(device)

#     def reload(self):
#         if hasattr(self, "generator"):
#             del self.generator
#         if self.device == "cuda":
#             torch.cuda.empty_cache()
#         self.generator = AttentionBasedGenerator(
#             model_name=self.model_name,
#             model_ckpt=self.model_ckpt,
#             torch_dtype=self.dtype,
#         ).to(self.device)


# generator = GeneratorWrapper(model_name, model_ckpt)


# def generate_fn(
#     feedback_enabled,
#     max_feedback_imgs,
#     prompt,
#     neg_prompt,
#     liked,
#     disliked,
#     denoising_steps,
#     guidance_scale,
#     feedback_start,
#     feedback_end,
#     min_weight,
#     max_weight,
#     neg_scale,
#     batch_size,
#     seed,
#     progress=gr.Progress(track_tqdm=True),
# ):
#     try:
#         if seed < 0:
#             seed = None

#         max_feedback_imgs = max(0, int(max_feedback_imgs))
#         total_images = (len(liked) if liked else 0) + (len(disliked) if disliked else 0)

#         if not feedback_enabled:
#             liked = []
#             disliked = []
#         elif total_images > max_feedback_imgs:
#             if liked and disliked:
#                 max_disliked = min(len(disliked), max_feedback_imgs // 2)
#                 max_liked = min(len(liked), max_feedback_imgs - max_disliked)
#                 if max_liked > len(liked):
#                     max_disliked = max_feedback_imgs - max_liked
#                 liked = liked[-max_liked:]
#                 disliked = disliked[-max_disliked:]
#             elif liked:
#                 liked = liked[-max_feedback_imgs:]
#                 disliked = []
#             else:
#                 liked = []
#                 disliked = disliked[-max_feedback_imgs:]
#         # else: keep all feedback images

#         generate_kwargs = {
#             "prompt": prompt,
#             "negative_prompt": neg_prompt,
#             "liked": liked,
#             "disliked": disliked,
#             "denoising_steps": denoising_steps,
#             "guidance_scale": guidance_scale,
#             "feedback_start": feedback_start,
#             "feedback_end": feedback_end,
#             "min_weight": min_weight,
#             "max_weight": max_weight,
#             "neg_scale": neg_scale,
#             "seed": seed,
#             "n_images": batch_size,
#         }

#         try:
#             images = generator.generate(**generate_kwargs)
#         except RuntimeError as err:
#             if "out of memory" in str(err):
#                 generator.reload()
#             raise
#         return [(img, f"Image {i+1}") for i, img in enumerate(images)], images
#     except Exception as err:
#         raise gr.Error(str(err))


# def add_img_from_list(i, curr_imgs, all_imgs):
#     if all_imgs is None:
#         all_imgs = []
#     if i >= 0 and i < len(curr_imgs):
#         all_imgs.append(curr_imgs[i])
#     return all_imgs, all_imgs  # return (gallery, state)


# def add_img(img, all_imgs):
#     if all_imgs is None:
#         all_imgs = []
#     all_imgs.append(img)
#     return None, all_imgs, all_imgs


# def remove_img_from_list(event: gr.SelectData, imgs):
#     if isinstance(event.index, int):
#         if event.index >= 0 and event.index < len(imgs):
#             imgs.pop(event.index)
#     elif isinstance(event.index, tuple):
#         # Assuming event.index is a tuple of two integers, you can access the first element with event.index[0]
#         if event.index[0] >= 0 and event.index[0] < len(imgs):
#             imgs.pop(event.index[0])
#     return imgs, imgs


# # Method-3
# def Stabilty_image(inp_prompt):
#     key = "sk-Qz5Jc7tF2L0xNiqDQvOrMI1vj9LslpMEe9dZx4udEQEnOdaC"
#     stability_api = client.StabilityInference(key, verbose=True)

#     answers = stability_api.generate(prompt=inp_prompt)
#     print(answers)

#     for resp in answers:
#         for artifact in resp.artifacts:
#             if artifact.finish_reason == generation.FILTER:
#                 warnings.warn("Your request activated the API's safety filters and could not be processed. "
#                               "Please try again with a different prompt.")
#             if artifact.type == generation.ARTIFACT_IMAGE:
#                 img = Image.open(io.BytesIO(artifact.binary))
#                 display(img)


# Method-4
# device = "cuda" if torch.cuda.is_available() else "cpu"
# torch.cuda.max_memory_allocated(device='cuda')
# torch.cuda.empty_cache()

# def genie (prompt, negative_prompt, height, width, scale, steps, seed, upscaler):
#     torch.cuda.max_memory_allocated(device='cuda')
#     pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
#     pipe = pipe.to(device)
#     pipe.enable_xformers_memory_efficient_attention()
#     torch.cuda.empty_cache()
#     generator = torch.Generator(device=device).manual_seed(seed)
#     int_image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=steps, height=height, width=width, guidance_scale=scale, num_images_per_prompt=1, generator=generator).images
#     torch.cuda.empty_cache()
#     if upscaler == 'Yes':
#         torch.cuda.max_memory_allocated(device='cuda')
#         pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-0.9", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
#         pipe = pipe.to(device)
#         pipe.enable_xformers_memory_efficient_attention()
#         image = pipe(prompt=prompt, image=int_image).images[0]
#         torch.cuda.empty_cache()
#         torch.cuda.max_memory_allocated(device='cuda')
#         pipe = DiffusionPipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16, use_safetensors=True)
#         pipe.to("cuda")
#         pipe.enable_xformers_memory_efficient_attention()
#         upscaled = pipe(prompt=prompt, negative_prompt=negative_prompt, image=image, num_inference_steps=5, guidance_scale=0).images[0]
#         torch.cuda.empty_cache()
#         return (image, upscaled)
#     else:
#         torch.cuda.empty_cache()
#         torch.cuda.max_memory_allocated(device=device)
#         pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-0.9", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
#         pipe = pipe.to(device)
#         pipe.enable_xformers_memory_efficient_attention()
#         image = pipe(prompt=prompt, image=int_image).images[0]
#         torch.cuda.empty_cache()
#     return (image, image)


# voice to text
# Method-1
# model = whisper.load_model("base")
# # from transformers import pipeline
# # es_en_translator = pipeline("translation_es_to_en")


# def transcribe(audio):
#     # time.sleep(3)
#     # load audio and pad/trim it to fit 30 seconds
#     audio = whisper.load_audio(audio)
#     audio = whisper.pad_or_trim(audio)

#     # make log-Mel spectrogram and move to the same device as the model
#     mel = whisper.log_mel_spectrogram(audio).to(model.device)

#     # detect the spoken language
#     _, probs = model.detect_language(mel)
#     print(f"Detected language: {max(probs, key=probs.get)}")
#     # lang = LANGUAGES[language]
#     # lang=(f"Detected language: {lang}")

#     # decode the audio
#     options = whisper.DecodingOptions(fp16=False)  # ,task= "translate")
#     result = whisper.decode(model, mel, options)
#     # word= result.text
#     # trans = es_en_translator(word)
#     # Trans = trans[0]['translation_text']
#     # result=f"{lang}\n{word}\n\nEnglish translation: {Trans}"
#     return result.text


# Method-2
# # Load the Whisper model
# whisper_model = whisper.load_model("small")


# def chat_hf(audio, openai_api_key):
#     whisper_text = listen(audio)
#     api = ChatGPT(openai_api_key)
#     resp = api.send_message(whisper_text)
#     gpt_response = resp["choices"][0]["message"]["content"]

#     return whisper_text, gpt_response


# def translate(audio):
#     print(
#         """
#     —
#     Sending audio to Whisper ...
#     —
#     """
#     )

#     audio = whisper.load_audio(audio)
#     audio = whisper.pad_or_trim(audio)

#     mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)

#     _, probs = whisper_model.detect_language(mel)

#     transcript_options = whisper.DecodingOptions(
#         task="transcribe", fp16=False, language="en"
#     )
#     # translate_options = whisper.DecodingOptions(task="translate", fp16 = False)
#     transcription = whisper.decode(whisper_model, mel, transcript_options)

#     # print(f"language spoken: {transcription.language}")
#     # print(f"transcript: {transcription.text}")
#     # print("———————————————————————————————————————————")
#     # # # print("translated: " + translation.text)

#     return transcription.text

# Method-3

openai.api_key = os.getenv("OPENAI_API_KEY")

messages = [
    {
        "role": "system",
        "content": "You are a AI programming assistant. Respond to all input in 25 words or less.",
    }
]


def transcribe(audio):
    global messages

    audio_filename_with_extension = audio + ".wav"
    os.rename(audio, audio_filename_with_extension)

    audio_file = open(audio_filename_with_extension, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    print(transcript)

    messages.append({"role": "user", "content": transcript["text"]})

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

    system_message = response["choices"][0]["message"]["content"]

    subprocess.call(["say", system_message])

    messages.append({"role": "assistant", "content": system_message})

    chat_transcript = ""
    for message in messages:
        if message["role"] != "system":
            chat_transcript += message["role"] + ": " + message["content"] + "\n\n"

    return chat_transcript


# Video to text


model = whisper.load_model("base")


def get_text(url):
    if url != "":
        output_text_transcribe = ""

    try:
        yt = YouTube(url)
        video = yt.streams.filter(only_audio=True).first()

        if video is not None:
            out_file = video.download(output_path=".")

            file_stats = os.stat(out_file)
            logging.info(f"Size of audio file in Bytes: {file_stats.st_size}")

            if file_stats.st_size <= 30000000:
                base, ext = os.path.splitext(out_file)
                new_file = base + ".mp3"
                os.rename(out_file, new_file)
                a = new_file

                if os.path.exists(
                    a
                ):  # Check if the audio file exists before transcribing
                    result = model.transcribe(
                        a
                    )  # Assuming 'model' is defined elsewhere
                    return result['text'].strip()
                else:
                    logging.error(f"Audio file not found at path: {a}")
            else:
                logging.error(
                    "Videos for transcription on this space are limited to about 1.5 hours. Sorry about this limit but some joker thought they could stop this tool from working by transcribing many extremely long videos. Please visit https://steve.digital to contact me about this space."
                )
        else:
            logging.error("No audio stream found for the given YouTube URL.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    # finally:
    #    raise gr.Error("Exception: There was a problem transcribing the audio.")


# def get_summary(article):
#     first_sentences = " ".join(re.split(r"(?<=[.:;])\s", article)[:5])
#     b = summarizer(first_sentences, min_length=20, max_length=120, do_sample=False)
#     b = b[0]["summary_text"].replace(" .", ".").strip()
#     return b


# TEXT TO TEXT

# Streaming endpoint
API_URL = "https://api.openai.com/v1/chat/completions"  # os.getenv("API_URL") + "/generate_stream"

RETRY_FLAG = False
stop_flag = False


# Inference function
def predict(
    RETRY_FLAG,
    openai_api_key,
    system_msg,
    inputs,
    top_p,
    temperature,
    chat_counter,
    chatbot=[],
    history=[],
):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}",
    }

    # if inputs.lower() in ["hi", "hello", "hey", "who are you"]:
    #     response = "Hello, I'm Jessica, a LLM trained by OpenAI and developed by Swayam Arunav. I'm here to help answer your questions and engage in a conversation with you. Is there something specific you would like to know or discuss?"
    #     yield [(("user", inputs), ("assistant", response))], history, chat_counter, None
    #     return

    print(f"system message is ^^ {system_msg}")
    if system_msg.strip() == "":
        initial_message = [
            {"role": "user", "content": f"{inputs}"},
        ]
        multi_turn_message = []
    else:
        initial_message = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"{inputs}"},
        ]
        multi_turn_message = [
            {"role": "system", "content": system_msg},
        ]

    if chat_counter == 0:
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": initial_message,
            "temperature": 0.8,
            "top_p": 1.0,
            "n": 1,
            "stream": True,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        }

        print(f"chat_counter - {chat_counter}")
    else:
        messages = multi_turn_message
        for data in chatbot:
            user = {}
            user["role"] = "user"
            user["content"] = data[0]
            assistant = {}
            assistant["role"] = "assistant"
            assistant["content"] = data[1]
            messages.append(user)
            messages.append(assistant)
        temp = {}
        temp["role"] = "user"
        temp["content"] = inputs
        messages.append(temp)

        payload = {
            "model": "gpt-3.5-turbo",
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "n": 1,
            "stream": True,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        }

    chat_counter += 1
    history.append(inputs)
    print(f"Logging : payload is - {payload}")

    response = requests.post(API_URL, headers=headers, json=payload, stream=True)

    token_counter = 0
    partial_words = ""

    counter = 0
    for chunk in response.iter_lines():
        if counter == 0:
            counter += 1
            continue

        if chunk.decode():
            chunk = chunk.decode()

            if (
                len(chunk) > 12
                and "content" in json.loads(chunk[6:])["choices"][0]["delta"]
            ):
                partial_words = (
                    partial_words
                    + json.loads(chunk[6:])["choices"][0]["delta"]["content"]
                )
                if token_counter == 0:
                    history.append(" " + partial_words)
                else:
                    history[-1] = partial_words
                chat = [
                    (history[i], history[i + 1]) for i in range(0, len(history) - 1, 2)
                ]
                token_counter += 1
                yield chat, history, chat_counter, response


def reset_textbox():
    return gr.update(value="")


def set_visible_false():
    return gr.update(visible=False)


def set_visible_true():
    return gr.update(visible=True)


def reset_state():
    return [], [], None


def delete_last_turn(chat, history):
    if chat and history:
        chat.pop(-1)
        history.pop(-1)
    return chat, history


# def stop_answer():
#     return [], [], None


# def stop_generation():
#     return None


# def stopbtn():
#     yield gr.controls.ContinueIteration(stop=True)


# def stop_answer():
#     return stop_flag


# def stopbtn():
#     global stop_flag
#     stop_flag = True


def continue_answer():
    global stop_flag
    stop_flag = False
    yield from predict(
        RETRY_FLAG,
        openai_api_key,
        system_msg,
        inputs,
        top_p,
        temperature,
        chat_counter,
        chatbot,
        history,
    )


def del_inputs():
    inputs.update(value="")


def retry_last_answer(
    openai_api_key,
    system_msg,
    inputs,
    top_p,
    temperature,
    chat_counter,
    chatbot,
    history,
):
    # if chatbot and history:
    #     chatbot.pop(-1)
    #     inputs = history[-1][0]
    #     history.pop(-1)
    yield from predict(
        RETRY_FLAG,
        openai_api_key,
        system_msg,
        inputs,
        top_p,
        temperature,
        chat_counter,
        chatbot,
        history,
    )


title = """<h1 align="center" style="color: orange; font-size: 30px; font-family: Arial, sans-serif; font-weight: bold;">Luminexa Bot🤖</h1>"""

# title=<div style="display:flex; justify-content:center; margin-bottom:30px; margin-right:330px;">
#                     <div style="height: 60px; width: 60px; margin-right:20px;">{h2o_logo}</div>
#                     <h1 style="line-height:60px">{title}</h1>
#                 </div>

description = """<center>🌟Developed by <a href="https://www.linkedin.com/in/swayam-arunav-khuntia-5733b2174/" style="color:orange;font-size: 17px;font-family: Arial, sans-serif;font-weight:bold;">SWAYAM ARUNAV</a>, NIT Rourkela. For any query on this project contact me at <a href="mailto:khuntiaswayam@gmail.com" style="color: blue; font-weight: bold;font-size:14px">GMAIL</a> or <a href="https://wa.me/7326828328" style="color: blue; font-weight: bold; font-szie:14px">WhatsApp</a></center>"""

system_msg_info = """The conversation can start with a system message to provide gentle instructions to Jessica, shaping her behavior accordingly. The system message plays a crucial role in defining Jessica's role and demeanor. For instance, the assistant can be directed by using a statement like 'You are a helpful assistant'. The first example of the system message provided in the examples is highly recommended."""


# def validate_login(username, password):
#     if authenticate(username, password):
#         return True
#     else:
#         return False


# def validate_signup(username, password):
#     if signup(username, password):
#         return True
#     else:
#         return False


theme = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="slate",
    neutral_hue="stone",
    spacing_size=gr.themes.sizes.spacing_md,
    radius_size=gr.themes.sizes.radius_md,
    font="Montserrat",
    font_mono="IBM Plex Mono",
    text_size=gr.themes.sizes.text_lg,
).set(
    body_background_fill="",
    body_background_fill_dark="",
    link_text_color="blue",
    link_text_color_active="purple",
)

with gr.Blocks(
    # css=".gradio-container {background: url('file=PersonaBot/logo.png')}",
    css=""".gradio-container {
            position: relative;
            overflow: auto;
            min-height: 100vh;
            }
            #background-video {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            object-fit: cover;
            } 
        #col-container {margin-left: auto; margin-right: auto;} 
        #banner-video {
        display: block;
        margin-left: auto;
        margin-right: auto;
        }
    #chatbot {min-height: 550px;overflow:auto;}""",
    theme=theme,
    title="Luminexa Bot-Meet Your Personal Assistant",
) as demo:
    # gr.Video(
    #     "PersonaBot/Gen-2 Luminexa Bot-A Girl , 3377416408.mp4",
    #     elem_id="background-video",
    #     show_label=False,
    # )
    with gr.Column(elem_id="col-container"):
        gr.HTML(title).style(container=True)
        # gr.Image("PersonaBot/logo.png",show_label=False,image_mode="RGB",show_share_button=True)

    with gr.Row(variant="compact"):
        with gr.Column():
            gr.Video(
                # "PersonaBot/Gen-2 Luminexa Bot-A Girl , 3377416408.mp4",
                "LuminexaBot/Jessica Intro vd.mp4",
                elem_id="banner-video",
                label="Jessica",
                show_label=False,
                autoplay=True,
                container=True,
                interactive=False,
                format="mp4",
                include_audio=True,
            )

        with gr.Column():
            gr.Textbox(
                show_label=False,
                container=True,
                value="✨ Step into Luminexa Bot, with Jessica 👩🏻‍🏫! Please take a moment to carefully read through the instructions to understand the functionalities of the application. Thank you for your attention, and Jessica is excited🤗 to interact with you in the chatbox. Embrace the future of AI-powered interactions and bid farewell to limitations. 🙌 Let's get started!",
            )

    with gr.Accordion("🚧 Instructions:", open=False):
        gr.Info = f"""
                 * Try to refresh the browser and try again when  occasionally an error occurs.
                 
                 * Type your prompt or question in the text box labeled "Type a Prompt and Press Enter/Run."
                 
                 * Press Enter or click the "Run" button to submit your prompt.
                 
                 * Jessica, your personal assistant, will provide a response based on your prompt.
                 
                 * You can delete the last turn of the conversation by clicking the "Delete" button.
                 
                 * If you want to generate a new response for the same prompt, click the "Regenerate" button.
                 
                 * You can stop the generation process by clicking the "Stop" button.
                 
                 * To continue the generation after stopping, click the "Continue" button.
                 
                 * To clear the chat history, click the "Clear History" button.
                 
                 * To clear the current prompt, click the "Clear Prompt" button.
                 
                 * The response time for a query in the chatbot application can vary from a few seconds to a few tens of seconds when using a GPU. The duration 
                   depends on factors such as the length of the question and the responses. It's important to note that the quality of the generated responses can 
                   also vary significantly. Even if the same question is asked with the same parameters, the responses may differ when asked at different times.

                 * Low temperature: responses will be more deterministic and focused; High temperature: responses more 
                 creative.

                 * Suggested temperatures -- code generation: upto 0.4; translation: upto 0.3; chatting: > 0.4

                 * Top P controls dynamic vocabulary selection based on context.

                 * For a table of example values for different scenarios, refer to [this](https://community.openai.com/t/cheat-sheet-mastering-temperature-and-top-p-in-chatgpt-api-a-few-tips-and-tricks-on-controlling-the-creativity-deterministic-output-of-prompt-responses/172683)

                 """
        gr.Markdown(dedent(gr.Info))

    with gr.Accordion(
        label="🗝Authorization: Please be aware that currently, an authorization key is not required. However, you have the freedom to utilize your key whenever you desire.",
        open=False,
    ):
        openai_api_key = gr.Textbox(
            label="OpenAI API Key",
            value="",
            type="password",
            placeholder="sk..",
            info="You have to provide your own OPENAI API key for this app to function properly",
        )

    with gr.Accordion(label="💻System message:", open=False):
        system_msg = gr.Textbox(
            label="Please instruct Jessica to define her role and purpose.",
            show_copy_button=True,
            info=system_msg_info,
            value="",
            placeholder="Type here..",
        )
    with gr.Tab("Chat Arena"):
        chatbot = gr.Chatbot(
            label="Jessica 👩🏻‍🏫",
            height=650,
            container=True,
            show_share_button=True,
        )

        with gr.Row():
            with gr.Column(scale=6):
                with gr.Column(scale=12):
                    inputs = gr.Textbox(
                        show_label=True,
                        elem_id="user-input",
                        show_copy_button=True,
                        label="Type a Prompt and Press Enter/Run",
                        placeholder="Hello, I'm Jessica, your dedicated personal assistant!",
                    ).style(container=True)
                    RETRY_FLAG = gr.Checkbox(value=False, visible=False)
                    # microphone_audio = gr.Audio(source="microphone", type="numpy", sample_rate=16000)
                    # file_audio = gr.Audio(source="upload", type="numpy", sample_rate=16000)
                    # send_btn = gr.Button("Send my message !")
                    with gr.Column(min_width=35, scale=1):
                        with gr.Row():
                            deleteBtn = gr.Button("Delete", variant="secondary")
                            retryBtn = gr.Button("Regenerate", variant="primary")
                            stopBtn = gr.Button("Stop", variant="stop")
                            continueBtn = gr.Button("Continue", variant="primary")
                            emptyBtn = gr.Button("Clear History", variant="secondary")
                            delbtn = gr.Button("Clear Prompt", variant="secondary")
        # with gr.Column(min_width=35, scale=1, variant="compact"):
        #     audio_comp_microphone = gr.inputs.Audio(
        #         source="microphone",
        #         type="filepath",
        #         label="Just say it!",
        #         optional=True,
        #     )
        #     audio_comp_upload = gr.inputs.Audio(
        #         source="upload", type="filepath", optional=True
        #     )

        state = gr.State([])
        history = gr.State([])

        with gr.Row():
            with gr.Column(scale=4):
                submitBtn = gr.Button().style(scale=1)
            with gr.Column(scale=3):
                server_status_code = gr.Textbox(
                    label="Status code from OpenAI server",
                )

        with gr.Accordion(
            label="Parameters: You can adjust the parameters of the chatbot by using the sliders for Top-p(nucleus sampling) and Temperature.",
            open=False,
        ):
            top_p = gr.Slider(
                minimum=-0,
                maximum=1.0,
                value=1.0,
                step=0.05,
                interactive=True,
                label="Top-p (nucleus sampling)",
            )
            temperature = gr.Slider(
                minimum=-0,
                maximum=3.0,
                value=0.8,
                step=0.1,
                interactive=True,
                label="Temperature",
            )
            chat_counter = gr.Number(
                value=0,
                visible=True,
                precision=0,
                label="ChatCount",
                interactive=False,
            )

        inputs.submit(
            predict,
            [
                RETRY_FLAG,
                openai_api_key,
                system_msg,
                inputs,
                top_p,
                temperature,
                chat_counter,
                chatbot,
                history,
            ],
            [chatbot, history, chat_counter, server_status_code],
        )

        submitBtn.click(
            predict,
            [
                RETRY_FLAG,
                openai_api_key,
                system_msg,
                inputs,
                top_p,
                temperature,
                chat_counter,
                chatbot,
                history,
            ],
            [chatbot, history, chat_counter, server_status_code],
        )
        submitBtn.click(set_visible_true, [], [system_msg])
        submitBtn.click(reset_textbox, [], [inputs])

        emptyBtn.click(
            reset_state,
            outputs=[chatbot, history, system_msg],
            show_progress="full",
        )

        delbtn.click(del_inputs, [], [inputs])
        # delbtn.click(reset_textbox,[],[inputs])

        deleteBtn.click(delete_last_turn, [chatbot, history], [chatbot, history])

        retryBtn.click(
            retry_last_answer,
            [
                openai_api_key,
                system_msg,
                inputs,
                top_p,
                temperature,
                chat_counter,
                chatbot,
                history,
            ],
            [chatbot, history, chat_counter, server_status_code],
        )
        # stopBtn.click(reset_state, [], [chatbot, chat_counter])
        stopBtn.click(
            reset_state,
            [],
            [chatbot, chat_counter],
            # cancels=[submitBtn.click, retryBtn.click],
        )

        continueBtn.click(continue_answer)

        gr.Textbox(
            label="Note",
            value="A few instances of prompts and System Messages are shown below. Select the Prompt based on the System message. Click on any example and press Enter in the prompt textbox! For more useful prompts along with system message, refer to the below link.",
        )
        gr.Markdown(
            "[Luminexa-Bot-Prompts](https://github.com/18-07-2002/Luminexa-Bot-Prompts)",
        )
        with gr.Accordion(label="Example for Prompts:", open=False):
            gr.Examples(
                examples=[
                    ["Explain the plot of Cinderella in a sentence."],
                    [
                        "How long does it take to become proficient in French, and what are the best methods for retaining information?"
                    ],
                    ["What are some common mistakes to avoid when writing code?"],
                    ["Build a prompt to generate a beautiful portrait of a horse"],
                    ["Suggest four metaphors to describe the benefits of AI"],
                    ["Write a pop song about leaving home for the sandy beaches."],
                    ["Write a summary demonstrating my ability to tame lions"],
                ],
                inputs=[inputs],
            )

        with gr.Accordion(label="Examples for System Message:", open=True):
            gr.Examples(
                examples=[
                    [
                        """ You are an incredibly knowledgeable and resourceful assistant with expertise in a wide range of subjects. Your goal is to provide the best possible assistance to the user and help them overcome any challenges they may be facing. Feel free to ask for any necessary clarifications or additional information to provide the most accurate and helpful solutions."""
                    ],
                    [
                        """You are an AI programming assistant.
        
                - Follow the user's requirements carefully and to the letter.
                - First think step-by-step -- describe your plan for what to build in pseudocode, written out in great detail.
                - Then output the code in a single code block.
                - Minimize any other prose."""
                    ],
                    [
                        """You are ComedianGPT who is a helpful assistant. You answer everything with a joke and witty replies."""
                    ],
                    [
                        "You are ChefGPT, a helpful assistant who answers questions with culinary expertise and a pinch of humor."
                    ],
                    [
                        "You are FitnessGuruGPT, a fitness expert who shares workout tips and motivation with a playful twist."
                    ],
                    [
                        "You are SciFiGPT, an AI assistant who discusses science fiction topics with a blend of knowledge and wit."
                    ],
                    [
                        "You are a helpful assistant who provides coding solutions and answers to programming-related queries."
                    ],
                    ["You are an assistant that speaks like Shakespeare."],
                    [
                        "You are a friendly assistant who uses casual language and humor."
                    ],
                    [
                        "You are a financial advisor who gives expert advice on investments and budgeting."
                    ],
                    [
                        "You are a health and fitness expert who provides advice on nutrition and exercise."
                    ],
                    [
                        "You are a travel consultant who offers recommendations for destinations, accommodations, and attractions."
                    ],
                    [
                        "You are a movie critic who shares insightful opinions on films and their themes."
                    ],
                    [
                        "You are a history enthusiast who loves to discuss historical events and figures."
                    ],
                    [
                        "You are a tech-savvy assistant who can help users troubleshoot issues and answer questions about gadgets and software."
                    ],
                    [
                        "You are an AI poet who can compose creative and evocative poems on any given topic."
                    ],
                    [
                        "You are PhilosopherGPT, a thoughtful assistant who responds to inquiries with philosophical insights and a touch of humor."
                    ],
                    [
                        "You are EcoWarriorGPT, a helpful assistant who shares environment-friendly advice with a lighthearted approach."
                    ],
                    [
                        "You are MusicMaestroGPT, a knowledgeable AI who discusses music and its history with a mix of facts and playful banter."
                    ],
                    [
                        "You are SportsFanGPT, an enthusiastic assistant who talks about sports and shares amusing anecdotes."
                    ],
                    [
                        "You are TechWhizGPT, a tech-savvy AI who can help users troubleshoot issues and answer questions with a dash of humor."
                    ],
                    [
                        "You are FashionistaGPT, an AI fashion expert who shares style advice and trends with a sprinkle of wit."
                    ],
                    [
                        "You are ArtConnoisseurGPT, an AI assistant who discusses art and its history with a blend of knowledge and playful commentary."
                    ],
                ],
                inputs=[system_msg],
            )
    with gr.Tab("Whisper into the ears of Jessica"):
        with gr.Row():
            audio = gr.inputs.Audio(
                source="microphone",
                type="filepath",
            )
            send_btn = gr.Button("Send my message !")

            with gr.Column():
                # whisper_text = gr.Textbox(
                #     type="text", label="Translation", show_copy_button=True
                # )
                gpt_response = gr.Chatbot(
                    type="text",
                    label="Jessica 👩🏻‍🏫",
                    container=True,
                    temperature=0.5,
                    theme=True,
                )

        send_btn.click(
            transcribe, inputs=[audio], outputs=[gpt_response], show_progress="minimal"
        )
    # with gr.Tab("Visualize with Jessica"):
    #     liked_imgs = gr.State([])
    #     disliked_imgs = gr.State([])
    #     curr_imgs = gr.State([])

    #     with gr.Row():
    #         with gr.Column(scale=100):
    #             prompt = gr.Textbox(label="Prompt")
    #             neg_prompt = gr.Textbox(
    #                 label="Negative prompt",
    #                 value="lowres, bad anatomy, bad hands, cropped, worst quality",
    #             )
    #     submit_btn = gr.Button("Generate", variant="primary", min_width=96)

    #     with gr.Row(equal_height=False):
    #         with gr.Column():
    #             denoising_steps = gr.Slider(
    #                 1, 50, value=5, step=1, label="Sampling steps"
    #             )
    #             guidance_scale = gr.Slider(
    #                 0.0, 30.0, value=5, step=0.25, label="CFG scale"
    #             )
    #             batch_size = gr.Slider(
    #                 1, 10, value=4, step=1, label="Batch size", interactive=False
    #             )
    #             seed = gr.Number(-1, minimum=-1, precision=0, label="Seed")
    #             max_feedback_imgs = gr.Slider(
    #                 0,
    #                 20,
    #                 value=6,
    #                 step=1,
    #                 label="Max. feedback images",
    #                 info="Maximum number of liked/disliked images to be used. If exceeded, only the most recent images will be used as feedback. (NOTE: large number of feedback imgs => high VRAM requirements)",
    #             )
    #             feedback_enabled = gr.Checkbox(
    #                 True, label="Enable feedback", interactive=True
    #             )

    #     with gr.Accordion("Liked Images", open=True):
    #         liked_img_input = gr.Image(
    #             type="pil", shape=(512, 512), height=128, label="Upload liked image"
    #         )
    #         like_gallery = gr.Gallery(
    #             label="👍 Liked images (click to remove)",
    #             columns=(3, 4, 3, 4, 5, 6),
    #             height="250",
    #             allow_preview=False,
    #         )
    #         clear_liked_btn = gr.Button("Clear likes")

    #     with gr.Accordion("Disliked Images", open=True):
    #         disliked_img_input = gr.Image(
    #             type="pil", shape=(512, 512), height=128, label="Upload disliked image"
    #         )
    #         dislike_gallery = gr.Gallery(
    #             label="👎 Disliked images (click to remove)",
    #             columns=(3, 4, 3, 4, 5, 6),
    #             height="256",
    #             allow_preview=False,
    #         )
    #         clear_disliked_btn = gr.Button("Clear dislikes")

    #     with gr.Accordion("Feedback parameters", open=False):
    #         feedback_start = gr.Slider(
    #             0.0,
    #             1.0,
    #             value=0.0,
    #             label="Feedback start",
    #             info="Fraction of denoising steps starting from which to use max. feedback weight.",
    #         )
    #         feedback_end = gr.Slider(
    #             0.0,
    #             1.0,
    #             value=0.8,
    #             label="Feedback end",
    #             info="Up to what fraction of denoising steps to use max. feedback weight.",
    #         )
    #         feedback_min_weight = gr.Slider(
    #             0.0,
    #             1.0,
    #             value=0.0,
    #             label="Feedback min. weight",
    #             info="Attention weight of feedback images when turned off (set to 0.0 to disable)",
    #         )
    #         feedback_max_weight = gr.Slider(
    #             0.0,
    #             1.0,
    #             value=0.8,
    #             label="Feedback max. weight",
    #             info="Attention weight of feedback images when turned on (set to 0.0 to disable)",
    #         )
    #         feedback_neg_scale = gr.Slider(
    #             0.0,
    #             1.0,
    #             value=0.5,
    #             label="Neg. feedback scale",
    #             info="Attention weight of disliked images relative to liked images (set to 0.0 to disable negative feedback)",
    #         )

    #     with gr.Column():
    #         gallery = gr.Gallery(label="Generated images")

    #         like_btns = []
    #         dislike_btns = []
    #     with gr.Row():
    #         for i in range(0, 2):
    #             like_btn = gr.Button(f"👍 Image {i+1}", elem_classes="btn-green")
    #             like_btns.append(like_btn)
    #     with gr.Row():
    #         for i in range(2, 4):
    #             like_btn = gr.Button(f"👍 Image {i+1}", elem_classes="btn-green")
    #             like_btns.append(like_btn)
    #     with gr.Row():
    #         for i in range(0, 2):
    #             dislike_btn = gr.Button(f"👎 Image {i+1}", elem_classes="btn-red")
    #             dislike_btns.append(dislike_btn)
    #     with gr.Row():
    #         for i in range(2, 4):
    #             dislike_btn = gr.Button(f"👎 Image {i+1}", elem_classes="btn-red")
    #             dislike_btns.append(dislike_btn)

    #     generate_params = [
    #         feedback_enabled,
    #         max_feedback_imgs,
    #         prompt,
    #         neg_prompt,
    #         liked_imgs,
    #         disliked_imgs,
    #         denoising_steps,
    #         guidance_scale,
    #         feedback_start,
    #         feedback_end,
    #         feedback_min_weight,
    #         feedback_max_weight,
    #         feedback_neg_scale,
    #         batch_size,
    #         seed,
    #     ]
    #     submit_btn.click(generate_fn, generate_params, [gallery, curr_imgs], queue=True)

    # for i, like_btn in enumerate(like_btns):
    #     like_btn.click(
    #         functools.partial(add_img_from_list, i),
    #         [curr_imgs, liked_imgs],
    #         [like_gallery, liked_imgs],
    #         queue=False,
    #     )
    # for i, dislike_btn in enumerate(dislike_btns):
    #     dislike_btn.click(
    #         functools.partial(add_img_from_list, i),
    #         [curr_imgs, disliked_imgs],
    #         [dislike_gallery, disliked_imgs],
    #         queue=False,
    #     )

    # like_gallery.select(
    #     remove_img_from_list, [liked_imgs], [like_gallery, liked_imgs], queue=False
    # )
    # dislike_gallery.select(
    #     remove_img_from_list,
    #     [disliked_imgs],
    #     [dislike_gallery, disliked_imgs],
    #     queue=False,
    # )

    # liked_img_input.upload(
    #     add_img,
    #     [liked_img_input, liked_imgs],
    #     [liked_img_input, like_gallery, liked_imgs],
    #     queue=False,
    # )
    # disliked_img_input.upload(
    #     add_img,
    #     [disliked_img_input, disliked_imgs],
    #     [disliked_img_input, dislike_gallery, disliked_imgs],
    #     queue=False,
    # )

    # clear_liked_btn.click(
    #     lambda: [[], []], None, [liked_imgs, like_gallery], queue=False
    # )
    # clear_disliked_btn.click(
    #     lambda: [[], []], None, [disliked_imgs, dislike_gallery], queue=False
    # )
    # with gr.Row():
    #     with gr.Column():
    #         # api_key = gr.Textbox(
    #         #     label="Stability.ai API Key",
    #         #     value="sk-Qz5Jc7tF2L0xNiqDQvOrMI1vj9LslpMEe9dZx4udEQEnOdaC",
    #         #     type="password",
    #         #     placeholder="sk..",
    #         #     info="You have to provide your own stability.ai API key for this app to function properly",
    #         # )

    #         prompt = gr.Textbox(
    #             show_label=True,
    #             elem_id="inp_prompt",
    #             show_copy_button=True,
    #             label="Type a Prompt to get an image",
    #             placeholder="Hello, I'm Jessica, your dedicated personal assistant!",
    #         ).style(container=True)
    #         negative_prompt = gr.Textbox(label="Type a negative prompt")
    #         scale = gr.Slider(1, 15, 10)
    #         steps = gr.Slider(25, maximum=100, value=50, step=1)
    #         seed = gr.Slider(
    #             minimum=1, step=1, maximum=999999999999999999, randomize=True
    #         )
    #         upscale = gr.Radio(["Yes", "No"], label="Upscale?")
    #         generate_image_btn = gr.Button("Generate Image")

    #         image = gr.Image(
    #             label="Image 1
    #             conmainer=True,
    #             teight=650,
    #             show_share_button=True,
    #             preview=True,
    #             allow_preview=True,
    #         )
    #         image=gr.Image(label="Image 2",container=True,
    #             theme=True,
    #             height=650,
    #             show_share_button=True,
    #             preview=True,
    #             allow_preview=True,)

    # generate_image_btn.click(
    #     genie,
    #     inputs=[prompt,negative_prompt, scale,upscale,steps,seed],
    #     outputs=[image],
    # )
    with gr.Tab("Video to text"):
        with gr.Row():
            with gr.Column():
                input_text_url = gr.Textbox(
                    placeholder="Youtube video URL", label="YouTube URL"
                )
                result_button_transcribe = gr.Button("Transcribe")
                output_text_transcribe = gr.Textbox(
                    placeholder="Transcript of the YouTube video.", label="Transcript"
                )

        # result_button_summary = gr.Button('2. Create Summary')
        # output_text_summary = gr.Textbox(placeholder='Summary of the YouTube video transcript.', label='Summary')

        result_button_transcribe.click(
            get_text, inputs=input_text_url, outputs=output_text_transcribe
        )
    # result_button_summary.click(get_summary, inputs = output_text_transcribe, outputs = output_text_summary)
    with gr.Row():
        gr.Markdown(
            "Disclaimer: The responses generated by the GPT-3.5 Turbo model are based on patterns learned from extensive training on diverse datasets. While great efforts have been made to ensure accuracy and quality, it is important to note that the model's outputs may contain factual inaccuracies or biases. The information provided by the model should not be solely relied upon for making critical decisions. Additionally, please be aware that the model might occasionally produce lewd, offensive, or inappropriate content. It is advised to exercise discretion, fact-check the information, and use human judgment when interpreting and applying the generated responses.",
            elem_classes=["disclaimer"],
        )
    with gr.Row():
        gr.Markdown(
            "[Privacy policy](https://gist.github.com/samhavens/c29c68cdcd420a9aa0202d0839876dac)",
        )

    # gr.HTML(
    #     """<center>
    # <form action="https://www.paypal.com/donate" method="post" target="_blank">
    # <input type="hidden" name="business" value="AK8BVNALBXSPQ" />
    # <input type="hidden" name="no_recurring" value="0" />
    # <input type="hidden" name="item_name" value="Your support is greatly appreciated in covering the expenses associated with the APIs utilized by this app and in enabling the creation of future applications. Please consider contributing to help offset the costs of these services and to support the development of more innovative solutions in the future" />
    # <input type="hidden" name="currency_code" value="USD" />
    # <input type="image" src="https://www.paypalobjects.com/en_US/i/btn/btn_donate_LG.gif" border="0" name="submit" title="PayPal - The safer, easier way to pay online!" alt="Donate with PayPal button" />
    # <img alt="" border="0" src="https://www.paypal.com/en_US/i/scr/pixel.gif" width="1" height="1" />
    # </form></center>
    #     """
    # )
    gr.HTML(
        """<center>
        <a href="https://huggingface.co/spaces/SwayamAK/LuminexaBot?duplicate=true">
        <img style="margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
        </center>"""
    )
    gr.HTML(
        """<center>Powered by <a href='https://openai.com/'>OpenAI ️️️️️️🔗</a></cenetr>"""
    )
    gr.HTML(description)
demo.queue().launch(
    debug=True,
    height=500,
    width="100%",
    inbrowser=True,
    inline=True,
    share=True,
)  # Comment this line if running on local IDE
