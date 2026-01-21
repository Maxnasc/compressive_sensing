import requests

def send_telegram_msg(message):
    token = "8384413791:AAEmVPO7oPWRAM9be8dQlKEtTdBKq-eiXX8"
    chat_id = "8555571604"
    url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={message}"
    try:
        requests.get(url)
    except Exception as e:
        print(f"Erro ao enviar notificação: {e}")

if __name__=="__main__":
    # Teste de funcionalidade da função send_telegram_msg
    send_telegram_msg("🚀 Bot Hermes entregando mensagens!")