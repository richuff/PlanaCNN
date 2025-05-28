import os
import ssh_refactor
import push_remote
from util.Logs.Log import Log
import time

if __name__ == '__main__':
    # Log.warning("初始化异常")
    # Log.error("参数错误")
    # Log.info("当前时间: " + str(time.time()))
    # Log.debug("参数值为: " + str(30))
    ssh_refactor.plana.setUser()
    ssh_refactor.plana.showUser()

    # ssh_refactor.richu.setUser()
    # ssh_refactor.richu.showUser()
    #
    # main = push_remote.PushRemote(os.getcwd(),"","origin","main")
    # main.git_push("test")
    #
    # main.refactor_branch("master")
