import subprocess
import webbrowser
import time

port = 1949

def start_server():
    command = f"python -m http.server {port}"
    process = subprocess.Popen(command, shell=True)
    return process

# Démarrer le serveur
process = start_server()

# Attendre un court instant pour que le serveur démarre
time.sleep(1)

# Ouvrir le fichier index.html dans un nouvel onglet
url = f"http://localhost:{port}/site/index.html"
new = 2  # open in a new tab, if possible
webbrowser.open(url, new=new)
print(f"Opening {url} in a new tab...")

# Garder le script en cours d'exécution
try:
    process.communicate()
except KeyboardInterrupt:
    process.terminate()
