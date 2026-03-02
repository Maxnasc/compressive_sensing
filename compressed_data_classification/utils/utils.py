"""
Module: utils.py

This module contains utility functions for the compressed data classification project.

Functions provide support for:
- Telegram notifications for training progress and completion messages

Author: Maxnasc7
License: MIT
"""

import os
from dotenv import load_dotenv
import requests

# Carrega as variáveis do arquivo .env
load_dotenv()

def send_telegram_msg(message):
    """
    Send a message to Telegram via Telegram Bot API.
    
    This function retrieves Telegram credentials from environment variables (.env file)
    and sends a message to a specified chat. Useful for notifying about training progress,
    completion, or errors.
    
    Parameters
    ----------
    message : str
        The message content to send via Telegram.
    
    Returns
    -------
    None
    
    Notes
    -----
    Requires the following environment variables to be set in .env file:
    - TELEGRAM_TOKEN: Bot token from BotFather
    - TELEGRAM_CHAT_ID: The chat ID where messages will be sent
    
    If credentials are not found, a warning message is printed and the function returns.
    """
    # Busca os valores das variáveis de ambiente
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        print("⚠️ Erro: Token ou Chat ID não encontrados no .env")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {'chat_id': chat_id, 'text': message}
    
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Erro ao enviar: {e}")

if __name__=="__main__":
    # Teste de funcionalidade da função send_telegram_msg
    send_telegram_msg("🚀 Bot Hermes entregando mensagens!")