import pyautogui as auto
import time

auto.hotkey('alt','tab')
# auto.press('home')

while True:
    auto.press('down')
    auto.press('enter')
    time.sleep(3)
    auto.hotkey('ctrl','s')
    auto.hotkey('alt','f4')



