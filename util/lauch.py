import os

import ssh_refactor
import push_remote

if __name__ == '__main__':
    # ssh_refactor.plana.setUser()
    # ssh_refactor.plana.showUser()

    ssh_refactor.richu.setUser()
    ssh_refactor.richu.showUser()

    # main = push_remote.PushRemote(os.getcwd(),"","origin","main")
    # main.git_push("test")
    #
    # main.refactor_branch("master")