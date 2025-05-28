import colorama

# 自动重置颜色
colorama.init(autoreset=True)

class Log:
    @staticmethod
    def warning(msg: str) -> None:
        """
        输出警告信息
        """
        print(f"{colorama.Fore.YELLOW}WARNING:{msg}",end="\n")

    @staticmethod
    def error(msg: str) -> None:
        """
        输出警告信息
        """
        print(f"{colorama.Fore.RED}ERROR:{msg}",end="\n")

    @staticmethod
    def info(msg: str) -> None:
        """
        输出打印信息
        """
        print(f"{colorama.Fore.GREEN}INFO:{msg}", end="\n")

    @staticmethod
    def debug(msg: str) -> None:
        """
        输出测试信息
        """
        print(f"{colorama.Fore.BLUE}DEBUG:{msg}", end="\n")

