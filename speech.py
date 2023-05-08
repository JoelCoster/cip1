import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
from pygame import mixer
import time
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
from hugchat import hugchat

chatbot = hugchat.ChatBot()

id = chatbot.new_conversation()
chatbot.change_conversation(id)

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

    tts = gTTS(text, lang='en')
    tts.write_to_fp(mp3_fp)

    mixer.init()
    mp3_fp.seek(0)
    mixer.music.load(mp3_fp, "mp3")
    mixer.music.play()

    while mixer.music.get_busy():
        time.sleep(0.1)


def dialogManagement(text):
    """
    Simple finite state dialog management for testing purposes. Takes an utterance as input and
    returns a textual response and a flag of wheter or not this ends the conversation
    """
    text = text.lower()
    triggers = ["talk to you later", "better luck next time", "see you"]
    for trigger in triggers:
        if trigger in text:
            return False, "Till next time"
        
    triggers = ["how are you", "how are you doing", "good how about you"]
    for trigger in triggers:
        if trigger in text:
            return True, "Mediocre at best"
        
    triggers = ["hello", "hey"]
    for trigger in triggers:
        if trigger in text:
            return True, "Hey how are you?"

    return True, chatbot.chat(text)

   

def turnManagement(start_conversation=True): #need conversation log as well
    if start_conversation:
        speak("Hello, I am Snoop")

    continue_conversation = True
    while continue_conversation:
        text = listen()
        if text:
            continue_conversation, response = dialogManagement(text)
            speak(response)


turnManagement()