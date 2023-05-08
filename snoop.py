import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import time
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
from datetime import datetime
from pygame import mixer
import sys
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import re

colorama_init()
global mic_index
mic_index = 0


def listen():
    stt = sr.Recognizer()

    with sr.Microphone(device_index = mic_index) as source:
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


class ChatLog:
    def __init__(self):
        self.log = []

    def addLine(self, participant, line):
        self.log.append([participant, line])

    def getLog(self):
        return self.log
    
    def getParticipantLog(self, participant):
        return [line for line in self.log if line[0] == participant]
    
    def exportCSV(self):
        exist = os.path.exists("logs")
        if not exist:
            os.makedirs("logs")

        filename = datetime.now().strftime("logs/%m-%d-%Y %H_%M_%S.csv")
        export = open(filename, "w")
        for line in self.log:
            export.write("\"" + str(line[0]) + "\",\"" + line[1].replace("\n", " ") + "\"\n")
        export.close()



class Snoop:
    def __init__(self):
        self.chatlog = ChatLog()
        #model_name = "facebook/blenderbot-400m-distill"
        #model_name = "facebook/blenderbot-1B-distill"
        model_name = "facebook/blenderbot-3B"
        self.model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = BlenderbotTokenizer.from_pretrained(model_name)


    def dialogManagement(self, text):
        text = text.lower()
        triggers = ["talk to you later", "exit"]
        for trigger in triggers:
            if trigger in text:
                return False, "Goodbye"

        inputs = self.tokenizer([text], return_tensors="pt")
        reply_ids = self.model.generate(**inputs, max_new_tokens=100)

        return True, re.sub(r'<.*?>', r'', self.tokenizer.batch_decode(reply_ids)[0]).strip()
   

    def run(self, start_conversation=True, speech=False):
        if start_conversation:
            initial_utterance = "Hello, I am Snoop"
            if speech: speak(initial_utterance)
            else: print(f"{Fore.MAGENTA}System: {Style.RESET_ALL}{initial_utterance}")
            self.chatlog.addLine("Snoop", initial_utterance)

        continue_conversation = True
        while continue_conversation:
            if speech: text = listen()
            else: text = input(f"{Fore.CYAN}User: {Style.RESET_ALL}")
            if text:
                self.chatlog.addLine("User", text)
                continue_conversation, response = self.dialogManagement(text)
                if speech: speak(response)
                else: print(f"{Fore.MAGENTA}System: {Style.RESET_ALL}{response}")
                self.chatlog.addLine("Snoop", response)

        self.chatlog.exportCSV()


if __name__ == "__main__":
    for arg in sys.argv:
        if "--mic" in arg: mic_index = int(arg[-1])

    snoop = Snoop()
    speech, start_converstation = False, False
    if "--speech" in sys.argv:
        speech = True
    if "--start" in sys.argv:
        start_converstation = True
        
    snoop.run(start_conversation=start_converstation, speech=speech)