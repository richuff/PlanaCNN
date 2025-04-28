import os


class config:
    def __init__(self, name, email):
        self.name = name
        self.email = email


plana = config("plana", "13058160568@163.com")
richu = config("richu", "1691401076@qq.com")


def setUser(config, config_path):
    os.system(f'git config --global user.email "{config.email}"')
    os.system(f'git config --global user.name "{config.name}"')
    with open(config_path, "w") as f:
        f.write("Host github.com\n")
        if config.name == "plana":
            f.write(f"\tIdentityFile ~/.ssh/{config.name}_id_rsa")
        elif config.name == "richu":
            f.write(f"\tIdentityFile ~/.ssh/id_rsa")


def showUser():
    print("git global UserName", end=": ")
    os.system('git config --global user.name')
    print("git global UserEmail", end=": ")
    os.system('git config --global user.email')
    print("\ncurrent ssh\config: ")
    with open(config_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            print(line, end="")


if __name__ == '__main__':
    username = os.getlogin()
    config_path = f"C:\\Users\\{username}\\.ssh\\config"
    setUser(richu, config_path)
    showUser()

# with open(config_path, "w+r") as f:
#     line = f.readline()
#     while line:
#         if "IdentityFile ~/.ssh/id_rsa" in line and config.name == "plana":
#             f.write(f"IdentityFile ~/.ssh/{config.name}_id_rsa")
#         elif "IdentityFile" in line and config.name == "richu":
#             f.write(f"IdentityFile ~/.ssh/id_rsa")
#         f.write(line)
#         line = f.readline()
