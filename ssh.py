import os


class config:
    def __init__(self, name, email,ssh_config_name="id_rsa"):
        self.name = name
        self.email = email
        self.ssh_config_name = ssh_config_name
        username = os.getlogin()
        self.config_path = f"C:\\Users\\{username}\\.ssh\\config"

    def setUser(self):
        os.system(f'git config --global user.email "{self.email}"')
        os.system(f'git config --global user.name "{self.name}"')
        with open(self.config_path, "w") as f:
            f.write("Host github.com\n")
            f.write(f"\tIdentityFile ~/.ssh/{self.ssh_config_name}")

    def showUser(self):
        print("git global UserName", end=": ")
        os.system('git config --global user.name')
        print("git global UserEmail", end=": ")
        os.system('git config --global user.email')
        print("\ncurrent ssh\config: ")
        with open(self.config_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                print(line, end="")


plana = config("plana", "13058160568@163.com","plana_id_rsa")
richu = config("richu", "1691401076@qq.com")

if __name__ == '__main__':
    # plana.setUser()
    # plana.showUser()
    richu.setUser()
    richu.showUser()








# with open(config_path, "w+r") as f:
#     line = f.readline()
#     while line:
#         if "IdentityFile ~/.ssh/id_rsa" in line and config.name == "plana":
#             f.write(f"IdentityFile ~/.ssh/{config.name}_id_rsa")
#         elif "IdentityFile" in line and config.name == "richu":
#             f.write(f"IdentityFile ~/.ssh/id_rsa")
#         f.write(line)
#         line = f.readline()
