import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
from pygame import mixer
import time
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
from hugchat import hugchat

colorama_init()


def listen():
    stt = sr.Recognizer()

    with sr.Microphone() as source:
        print(f"{Fore.GREEN}Listening...{Style.RESET_ALL}")
        stt.pause_threshold = 1
        audio = stt.listen(source)

    try:
        print(f"{Fore.GREEN}Recognizing...{Style.RESET_ALL}")
        query = stt.recognize_google(audio, language='en-us')
        print(f"{Fore.CYAN}User: {Style.RESET_ALL}{query}")

    except Exception as e:
        print(e)
        return None

    return query


def speak(text):
    mp3_fp = BytesIO()

    print(f"{Fore.MAGENTA}System: {Style.RESET_ALL}{text}")

    tts = gTTS(text, lang='en', tld='ca')
    tts.write_to_fp(mp3_fp)

    mixer.init()
    mp3_fp.seek(0)
    mixer.music.load(mp3_fp, "mp3")
    mixer.music.play()

    while mixer.music.get_busy():
        time.sleep(0.1)


class Snoop:
    def __init__(self):
        self.hugchat = hugchat.ChatBot()
        self.id = self.hugchat.new_conversation()
        self.hugchat.change_conversation(self.id)


    def dialogManagement(self, text):
        """
        Simple finite state dialog management for testing purposes. Takes an utterance as input and
        returns a textual response and a flag of wheter or not this ends the conversation
        """
        text = text.lower()
        triggers = ["talk to you later", "better luck next time", "see you"]
        for trigger in triggers:
            if trigger in text:
                return False, "Till next time"
       
        return True, self.hugchat.chat(text, truncate=400)
   

    def run(self, start_conversation=True, speech=False):
        if start_conversation:
            if speech: speak("Hello, I am Snoop")
            else: print("Hello, I am Snoop")

        continue_conversation = True
        while continue_conversation:
            if speech: text = listen()
            else: text = input("> ")
            if text:
                continue_conversation, response = self.dialogManagement(text)
                if speech: speak(response)
                else: print(response)


snoop = Snoop()
snoop.run()