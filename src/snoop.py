import os
import re
import sys
import time
from datetime import datetime
from io import BytesIO
from random import choice

import pandas as pd
import speech_recognition as sr
from colorama import Fore
from colorama import Style
from colorama import init as colorama_init
from gtts import gTTS
from pygame import mixer
from transformers import BlenderbotTokenizer, \
    BlenderbotForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

colorama_init()
mic_index = 0


def listen(i_participant, punct_tokenizer, punct_model):
    stt = sr.Recognizer()

    with sr.Microphone(device_index=mic_index) as source:
        print(f"{Fore.GREEN}Listening...{Style.RESET_ALL}")
        stt.pause_threshold = 1
        start_utter = datetime.now()
        audio = stt.listen(source)
    try:
        print(f"{Fore.GREEN}Recognizing...{Style.RESET_ALL}")
        query = stt.recognize_google(audio, language='en-us')

        inputs = punct_tokenizer.encode("punctuate: " + query,
                                        return_tensors="pt")
        result = punct_model.generate(inputs)

        query = punct_tokenizer.decode(result[0],
                                       skip_special_tokens=True)

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
        time.sleep(0.01)


class ChatLog:
    def __init__(self):
        self.id = 0
        self.log = {'time': [],
                    'speaker': [],
                    'utterance': []}

    def addLine(self, participant, line, utter_start):
        self.log['time'].append(str(utter_start))
        self.log['speaker'].append(participant)
        self.log['utterance'].append(line)

    def getLog(self):
        return self.log

    def getParticipantLog(self, participant):
        return [line for line in self.log if line[0] == participant]

    def exportCSV(self):
        exist = os.path.exists("logs")
        if not exist:
            os.makedirs("logs")

        filename = f"logs/{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.csv"
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

        self.punct_tokenizer = T5Tokenizer.from_pretrained(
            'SJ-Ray/Re-Punctuate')
        self.punct_model = T5ForConditionalGeneration.from_pretrained(
            'SJ-Ray/Re-Punctuate', from_tf=True)

    def dialogManagement(self, text):
        triggers = ["talk to you later", "end the conversation",
                    "end conversation"]
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

    def introduceTopic(self):
        topics_file = open("../assets/topics.txt", "r")
        topics = []
        for line in topics_file.readlines():
            topics.append(line.strip())
        return choice(topics)

    def startConversation(self):
        made_contact = False
        while not made_contact:
            utterance = choice(
                ["hello", "is anybody there", "hey", "who's there",
                 "greetings"])
            if speech:
                self.updateStartUtter()
                speak(utterance, self.o_participant)
            else:
                print(
                    f"{Fore.MAGENTA}{self.o_participant}: {Style.RESET_ALL}{utterance}")
            self.chatlog.addLine(self.o_participant, utterance,
                                 self.start_utter)

            if speech:
                text, start_utter = listen(self.i_participant,
                                           self.punct_tokenizer,
                                           self.punct_model)
            else:
                start_utter = datetime.now()
                text = input(
                    f"{Fore.CYAN}{self.i_participant}: {Style.RESET_ALL}")
            self.updateStartUtter(start_utter)

            if text:  # triggers should be loaded from txt file in assets
                triggers = ["hello", "hey", "how are you", "talking to me",
                            "who are you", "hi", "sup",
                            "what's up", "help,", "greetings", "salutations",
                            "morning", "afternoon",
                            "evening", "good day", "goodday", "i am"]
                for trigger in triggers:
                    if trigger in text.lower():
                        return text

    def run(self, start_conversation=True, speech=False, i='JAM', o='SNO'):
        try:
            self.i_participant = '{0}_IN'.format(i)
            self.o_participant = '{0}_OUT'.format(o)

            continue_conversation = True
            introduce_topic = False
            while continue_conversation:
                start_utter = datetime.now()
                if start_conversation:
                    text = self.startConversation()
                    start_conversation = False
                    if len(text.split()) < 6:
                        introduce_topic = True
                elif speech:
                    text, start_utter = listen(self.i_participant,
                                               self.punct_tokenizer,
                                               self.punct_model)
                else:
                    text = input(
                        f"{Fore.CYAN}{self.i_participant}: {Style.RESET_ALL}")
                self.updateStartUtter(start_utter)

                if text:
                    self.chatlog.addLine(self.i_participant, text,
                                         self.start_utter)
                    if not introduce_topic:
                        continue_conversation, response = self.dialogManagement(
                            text)
                    else:
                        introduce_topic = False
                        response = self.introduceTopic()
                    if speech:
                        self.updateStartUtter()
                        speak(response, self.o_participant)
                    else:
                        print(
                            f"{Fore.MAGENTA}{self.o_participant}: {Style.RESET_ALL}{response}")
                    self.chatlog.addLine(self.o_participant, response,
                                         self.start_utter)

            self.chatlog.exportCSV()

        except KeyboardInterrupt:
            self.chatlog.exportCSV()


if __name__ == "__main__":
    i, o = 'JAM', 'SNO'

    for arg in sys.argv:
        print(arg)
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
