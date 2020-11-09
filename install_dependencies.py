from subprocess import run

with open('install_dependencies.sh', encoding='utf-8-sig',mode='r') as file:
    for i,line in enumerate(file.readlines()):
        print(i)
        run(line, shell=True)