from subprocess import run

with open('install_dependencies.sh', encoding='utf-8-sig',mode='r') as file:
    for line in file.readlines():
        run(line, shell=True)