import pyautogui
import time

#pyautogui.PAUSE = 1


class GUI:
    """
    """

    def __init__(self, MOUSEDOWN = False, DOWN = False):
        self.MOUSEDOWN = MOUSEDOWN
        self.DOWN = DOWN

    def down(self):
            t1 = time.time()
            while True:
                pyautogui.click(1229, 681, duration=2)
                if time.time() - t1 > 3:
                    break

    def mousedown(self):
            t1 = time.time()
            while True:
                pyautogui.mouseDown(button='left')
                if time.time() - t1 > 3:
                    break

    def test(self):
        print(pyautogui.onScreen(pyautogui.position().x, pyautogui.position().y))


if __name__ == '__main__':
    gui = GUI(True, False)
    gui.down()
