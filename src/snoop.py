import os
import re
import sys
import time
from datetime import datetime
from io import BytesIO
from random import choice
from random import randint

import pandas as pd
import speech_recognition as sr
from colorama import Fore
from colorama import Style
from colorama import init as colorama_init
from gtts import gTTS
from pygame import mixer
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration
import warnings
import csv

warnings.filterwarnings("ignore")
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

        inputs = punct_tokenizer.encode("punctuate: " + query, return_tensors="pt")
        result = punct_model.generate(inputs)

        query = punct_tokenizer.decode(result[0], skip_special_tokens=True)

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


class PreviousConversations:
    def __init__(self):
        self.log = []

    def update(self, participant):
        for filename in os.listdir("logs/"):
            if filename[-3:] == "csv":
                f = open("logs/" + filename, "r")
                f.readline()
                for line in f.readlines():
                    try:
                        speaker = line.split(",")[2][1:-1]
                        utterance = ",".join(line.strip().split(",")[3:])[1:-1]
                        if speaker == participant:
                            self.log.append(utterance)
                    except:
                        pass

    def hasBeenSaidPreviously(self, utterance):
        repetition, question = False, False
        try:
            utterance = utterance.strip().split()
            for line in self.log:
                previous_utterance = line.strip().split()
                if utterance == previous_utterance:
                    if len(utterance) > 8:
                        repetition = True
                        if utterance[-1][-1] == "?":
                            question = True
            return repetition, question
        except:
            return repetition, question



class ChatLog:
    def __init__(self):
        self.id = 0
        self.log = {'time': [], 'speaker': [], 'utterance': []}

    def addLine(self, participant, line, utter_start):
        self.log['time'].append(str(utter_start))
        self.log['speaker'].append(participant)
        self.log['utterance'].append(line)

    def getLog(self):
        return self.log

    def getParticipantLog(self, participant):
        return [self.log["utterance"][i] for i in range(len(self.log["speaker"])) if self.log["speaker"][i] == participant]
    
    def hasBeenSaid(self, participant, utterance):
        repetition, question = False, False
        try:
            utterance = utterance.strip().split()
            for line in self.getParticipantLog(participant):
                previous_utterance = line.strip().split()
                if utterance == previous_utterance:
                    if len(utterance) > 8:
                        repetition = True
                    if utterance[-1][-1] == "?":
                        repetition, question = True, True
            return repetition, question
        except:
            return repetition, question

    def exportCSV(self):
        exist = os.path.exists("logs")
        if not exist:
            os.makedirs("logs")

        filename = f"logs/{datetime.now().isoformat(sep=' ', timespec='milliseconds')}".replace(":", "_").replace(".", "_") + ".csv"
        log_df = pd.DataFrame(self.log)
        log_df.to_csv(filename, index=True, quoting=csv.QUOTE_NONNUMERIC)


class Snoop:
    def __init__(self):
        self.chatlog = ChatLog()
        self.previous_conversations = PreviousConversations()
        model_name = "facebook/blenderbot-400m-distill"
        # model_name = "facebook/blenderbot-1B-distill"

        self.model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
        self.start_utter = datetime.now()
        self.i_participant = "User"
        self.o_participant = "System"

        self.punct_tokenizer = T5Tokenizer.from_pretrained('SJ-Ray/Re-Punctuate')
        self.punct_model = T5ForConditionalGeneration.from_pretrained('SJ-Ray/Re-Punctuate', from_tf=True)

    def dialogManagement(self, text):
        triggers = ["talk to you later", "end the conversation", "end conversation"]
        for trigger in triggers:
            if trigger in text.lower():
                return False, "Goodbye"

        inputs = self.tokenizer([text], return_tensors="pt")
        reply_ids = self.model.generate(**inputs)

        return True, re.sub(r'<.*?>', r'', self.tokenizer.batch_decode(reply_ids)[0]).strip()

    def updateStartUtter(self, start_utter=None):
        if start_utter:
            self.start_utter = start_utter
        else:
            self.start_utter = datetime.now()

    def introduceTopic(self):
        topics_file = open("assets/topics.txt", "r")
        topics = []
        transitions = ["Anyway, ", "By the way, ", "On another subject, ", "Anyway, ", "Anyway, "]
        for line in topics_file.readlines():
            topics.append(line.strip())
        return choice(transitions) + choice(topics)

    def startConversation(self):
        made_contact = False
        while not made_contact:
            utterance = choice(["hello", "hey", "greetings"])
            if speech:
                self.updateStartUtter()
                speak(utterance, self.o_participant)
            else:
                print(f"{Fore.MAGENTA}{self.o_participant}: {Style.RESET_ALL}{utterance}")
            self.chatlog.addLine(self.o_participant, utterance, self.start_utter)

            if speech:
                text, start_utter = listen(self.i_participant, self.punct_tokenizer, self.punct_model)
            else:
                start_utter = datetime.now()
                text = input(f"{Fore.CYAN}{self.i_participant}: {Style.RESET_ALL}")
            self.updateStartUtter(start_utter)

            if text:  # triggers should be loaded from txt file in assets
                triggers = ["hello", "hey", "how are you", "talking to me", "who are you", "hi", "sup",
                            "what's up", "help,", "greetings", "salutations", "morning", "afternoon",
                            "evening", "good day", "goodday", "i am", "yes", "nice to"]
                for trigger in triggers:
                    if trigger in text.lower():
                        return text
                    
            else:
                if speech:
                    text, start_utter = listen(self.i_participant, self.punct_tokenizer, self.punct_model)
                if text:  # triggers should be loaded from txt file in assets
                    triggers = ["hello", "hey", "how are you", "talking to me", "who are you", "hi", "sup",
                                "what's up", "help,", "greetings", "salutations", "morning", "afternoon",
                                "evening", "good day", "goodday", "i am"]
                    for trigger in triggers:
                        if trigger in text.lower():
                            return text

    def run(self, start_conversation=True, speech=False, i='JAM', o='SNO'):
        try:
            self.i_participant = '{0}_IN'.format(i)
            self.o_participant = '{0}_OUT'.format(o)
            self.previous_conversations.update(self.i_participant)

            continue_conversation = True
            introduce_topic = False
            while continue_conversation:
                response = ""
                start_utter = datetime.now()
                if start_conversation:
                    text = self.startConversation()
                    start_conversation = False
                    #if len(text.split()) < 8: introduce_topic = True # should probably be disabled for evaluation run
                elif speech:
                    text, start_utter = listen(self.i_participant, self.punct_tokenizer, self.punct_model)
                else:
                    text = input(f"{Fore.CYAN}{self.i_participant}: {Style.RESET_ALL}")
                self.updateStartUtter(start_utter)

                # check if user is repeating themselves
                repetition, question = self.chatlog.hasBeenSaid(self.i_participant, text)
                if repetition:
                    if question:
                        response = choice(["You already asked that. ", "didn't I just answer that? ", "Don't you remember I just told you? "])
                    else:
                        response = choice(["You already said that. ", "Yeah you told me. ", "I know, you told me that. "])

                # check if the user is repeating from a previous conversation
                if not repetition:
                    repetition, question = self.previous_conversations.hasBeenSaidPreviously(text)
                    if repetition:
                        if question:
                            response = choice(["I remember you asking that in an earlier conversation. ", "If I recall correctly you asked me that once before. "])
                        else:
                            response = choice(["I know, you told me in a previous conversation. ", "Yes, I remember that from earlier. ", "Right, I remember you telling me that once before. "])

                # introduce new topics
                if text and randint(1, 100) < 30 and not "?" in text and not repetition:
                    introduce_topic = True
                elif text and randint(1, 100) < 70 and repetition:
                    introduce_topic = True

                # regular responses
                if text:
                    self.chatlog.addLine(self.i_participant, text, self.start_utter)
                    if not introduce_topic and not repetition:
                        continue_conversation, response = self.dialogManagement(text)
                    else:
                        introduce_topic = False
                        response += self.introduceTopic()
                    if speech:
                        self.updateStartUtter()
                        speak(response, self.o_participant)
                    else:
                        print(f"{Fore.MAGENTA}{self.o_participant}: {Style.RESET_ALL}{response}")
                    self.chatlog.addLine(self.o_participant, response, self.start_utter)

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
