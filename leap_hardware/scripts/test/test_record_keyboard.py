#!/usr/bin/env python

import sys
import os
from select import select
if sys.platform == 'win32':
    import msvcrt
else:
    import termios
    import tty

import rospy
from std_msgs.msg import Bool
import time
import numpy as np

def getKey(settings, timeout):
    if sys.platform == 'win32':
        # getwch() returns a string on Windows
        key = msvcrt.getwch()
    else:
        tty.setraw(sys.stdin.fileno())
        # sys.stdin.read() returns a string on Linux
        rlist, _, _ = select([sys.stdin], [], [], timeout)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def saveTerminalSettings():
    if sys.platform == 'win32':
        return None
    return termios.tcgetattr(sys.stdin)

def main_record_keyboard():
    settings = saveTerminalSettings()
    lastkey = ""
    key_is_pressed = False
    total_receive_key = 0

    key_press_time = []
    key_release_time = []

    while not rospy.is_shutdown():
        key = getKey(settings, timeout=0.1)
        if key == "q" or key == "Q":
            break
        if key == " " and not key_is_pressed:
            print("Space key pressed")
            key_is_pressed = True
            key_press_time.append(rospy.Time.now().to_sec())
        elif key_is_pressed and key != " ":
            # print("Space key released")
            key_is_pressed = False
            key_release_time.append(rospy.Time.now().to_sec())

    for i in range(len(key_press_time)):
        print("Key pressed at {}!".format(key_press_time[i]))

    dirname = "/home/yongpeng/competition/leap_ws/src/leap_XL/leap_sim/leapsim/debug/fsd"
    idx = len([f for f in os.listdir(dirname) if f.endswith("npy")])
    np.save(os.path.join(dirname, f"key_press_time_{idx}.npy"), key_press_time)

if __name__ == '__main__':
    rospy.init_node('record_keyboard_node')
    main_record_keyboard()