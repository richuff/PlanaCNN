import os
import ssh_refactor
import push_remote
from util.Logs.Log import Log
import time

if __name__ == '__main__':
    if True:
        ssh_refactor.richu.changeUser()
    else:
        ssh_refactor.plana.changeUser()
    # main = push_remote.PushRemote(os.getcwd(),"","origin","main")
    # main.git_push("test")
    #
    # main.refactor_branch("master")
