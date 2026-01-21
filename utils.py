import os
from dotenv import load_dotenv
import requests

# Carrega as variáveis do arquivo .env
load_dotenv()

def send_telegram_msg(message):
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