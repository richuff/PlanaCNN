import os
class PushRemote:
    def __init__(self,path,remote_path,origin_name,branch_name):
        self.dir_name = path
        self.remote_path = remote_path
        self.origin_name = origin_name
        self.branch_name = branch_name
        self.add_remote()

    def git_push(self,message):
        os.system('git add .')
        os.system(f'git commit -m "{message}"')
        os.system(f'git push {self.origin_name} {self.branch_name}')
    def add_remote(self):
        os.system(f"git remote add origin {self.remote_path}")

    def remove_remote(self):
        os.system("git remote remove origin")

    def refactor_branch(self,new_branch_name):
        os.system(f"git branch -M {new_branch_name}")
