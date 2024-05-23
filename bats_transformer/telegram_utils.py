import requests
import os

'''
create a bot on telegram by contacting the user botfather, and get the bot token. 
create an environment variable TELEGRAM_BOT_TOKEN and assign the bot token to it.
create a chat on telegram and add the bot to the chat. next, add the CHAT_ID to 
the environment variable TELEGRAM_CHAT_ID. Then you can use the functions 
send_telegram_message and send_telegram_image to send messages and images to the chat.
'''
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"];
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"];


'''
send a telegram message.
'''      
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": message
    }
    response = requests.post(url, data=data)
    return response.json()

'''
send an image on telegram.
'''
def send_telegram_image(image_path):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    data = {
        "chat_id": CHAT_ID,
    }
    files = {
        "photo": open(image_path, "rb")
    }
    response = requests.post(url, data=data, files=files)
    return response.json()