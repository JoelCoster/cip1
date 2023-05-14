import os
import speech_recognition as sr
import time
import sys
import re
import pandas as pd

from transformers import BlenderbotTokenizer, \
    BlenderbotForConditionalGeneration
from gtts import gTTS
from io import BytesIO
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
from datetime import datetime
from pygame import mixer
from random import choice

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

colorama_init()
mic_index = 0


def listen(i_participant):
    stt = sr.Recognizer()

    with sr.Microphone(device_index=mic_index) as source:
        print(f"{Fore.GREEN}Listening...{Style.RESET_ALL}")
        stt.pause_threshold = 1
        start_utter = datetime.now()
        audio = stt.listen(source)
    try:
        print(f"{Fore.GREEN}Recognizing...{Style.RESET_ALL}")
        query = stt.recognize_google(audio, language='en-us')
        print(f"{Fore.CYAN}{i_participant}: {Style.RESET_ALL}{query}")

    except Exception as e:
        print(e)
        return None, None

    return query, start_utter



def speak(text, o_participant):
    mp3_fp = BytesIO()

    print(f"{Fore.MAGENTA}{o_participant}: {Style.RESET_ALL}{text}")

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
        self.id = 0
        self.log = {'time': [],
                    'speaker': [],
                    'utterance': []}


    def addLine(self, participant, line, utter_start):
        # Set time for line
        self.log['time'].append(str(utter_start))

        # Set speaker for line
        self.log['speaker'].append(participant)

        # Set utterance for line
        self.log['utterance'].append(line)


    def getLog(self):
        return self.log


    def getParticipantLog(self, participant):
        return [line for line in self.log if line[0] == participant]


    def exportCSV(self):
        exist = os.path.exists("logs")
        if not exist:
            os.makedirs("logs")

        filename = datetime.now().strftime("logs/%m-%d-%Y %H_%M_%S.csv")
        log_df = pd.DataFrame(self.log)
        log_df.to_csv(filename, index=True)



class Snoop:
    def __init__(self):
        self.chatlog = ChatLog()
        model_name = "facebook/blenderbot-400m-distill"
        # model_name = "facebook/blenderbot-1B-distill"
        # model_name = "facebook/blenderbot-3B"
        self.model = BlenderbotForConditionalGeneration.from_pretrained(
            model_name)
        self.tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
        self.start_utter = datetime.now()
        self.i_participant = "User"
        self.o_participant = "System"


    def dialogManagement(self, text):
        triggers = ["talk to you later", "end the conversation", "end conversation"]
        for trigger in triggers:
            if trigger in text.lower():
                return False, "Goodbye"

        inputs = self.tokenizer([text], return_tensors="pt")
        reply_ids = self.model.generate(**inputs, max_new_tokens=100)

        return True, re.sub(r'<.*?>', r'',
                            self.tokenizer.batch_decode(reply_ids)[0]).strip()


    def updateStartUtter(self, start_utter=None):
        if start_utter:
            self.start_utter = start_utter
        else:
            self.start_utter = datetime.now()


    def startConversation(self):
        made_contact = False
        while not made_contact:
            utterance = choice(["hello", "is anybody there", "hey"])
            if speech:
                self.updateStartUtter()
                speak(utterance, self.o_participant)
            else:
                print(f"{Fore.MAGENTA}{self.o_participant}: {Style.RESET_ALL}{utterance}")
            self.chatlog.addLine(self.o_participant, utterance, self.start_utter)
            
                
            if speech:
                text, start_utter = listen(self.i_participant)
            else:
                start_utter = datetime.now()
                text = input(f"{Fore.CYAN}{self.i_participant}: {Style.RESET_ALL}")
            self.updateStartUtter(start_utter)

            if text:
                triggers = ["hello", "hey", "how are you"]
                for trigger in triggers:
                    if trigger in text.lower():
                        self.chatlog.addLine(self.i_participant, text, self.start_utter)
                        return text


    def run(self, start_conversation=True, speech=False, i='JAM', o='SNO'):
        self.i_participant = '{0}_IN'.format(i)
        self.o_participant = '{0}_OUT'.format(o)

        continue_conversation = True
        while continue_conversation:
            start_utter = datetime.now()
            if start_conversation:
                text = self.startConversation()
                start_conversation = False
            elif speech:
                text, start_utter = listen(self.i_participant)
            else:
                text = input(f"{Fore.CYAN}{self.i_participant}: {Style.RESET_ALL}")
            self.updateStartUtter(start_utter)

            if text:
                self.chatlog.addLine(self.i_participant, text, self.start_utter)
                continue_conversation, response = self.dialogManagement(text)
                if speech:
                    self.updateStartUtter()
                    speak(response, self.o_participant)
                else:
                    print(
                        f"{Fore.MAGENTA}{self.o_participant}: {Style.RESET_ALL}{response}")
                self.chatlog.addLine(self.o_participant, response, self.start_utter)

        self.chatlog.exportCSV()



if __name__ == "__main__":
    i, o = 'JAM', 'SNO'

    for arg in sys.argv:
        if "--mic" in arg:
            mic_index = int(arg[-1])
        if "--i" in arg:
            i = arg.split('=')[-1]
        if "--o" in arg:
            o = arg.split('=')[-1]

    snoop = Snoop()
    speech, start_conversation = False, False
    if "--speech" in sys.argv:
        speech = True
    if "--start" in sys.argv:
        start_conversation = True

    snoop.run(start_conversation=start_conversation, speech=speech, i=i, o=o)