from selenium import webdriver
import cv2
import numpy as np
from imgaug import augmenters as iaa
import os
import string
import matplotlib.pyplot as plt


def save(path):
    a = cv2.imread(path, 0)
    a[a > 250] = 255
    a[a <= 250] = 0
    w = np.argmin(a, axis=1)
    w[w == 0] = 5000
    left = min(w)
    # print(left)
    h = np.argmin(a, axis=0)
    h[h == 0] = 5000
    top = min(h)
    a = np.rot90(a)

    # print(top)
    h = np.argmin(a, axis=0)
    h[h == 0] = 5000
    right = min(h)
    right = a.shape[0] - right
    # print(right)
    a = np.rot90(a)
    w = np.argmin(a, axis=0)
    w[w == 0] = 5000
    bottom = min(w)
    bottom = a.shape[0] - bottom
    # print(bottom)

    a = cv2.imread(path, 0)
    a = np.pad(
        a[top - 1 : bottom + 1, left - 1 : right + 1],
        (1, 1),
        "constant",
        constant_values=(255),
    )

    cv2.imwrite(path, a)
    return a


# main driver function to render html and take a screenshot


def generate_images():
    current_path = "/".join(os.path.abspath(__file__).split("/")[:-1])
    font_list = os.listdir(os.path.join(current_path, "./fonts/"))
    font_face = ""
    for i, j in enumerate(font_list):
        a = "@font-face { font-family: %s; src: url('./fonts/%s')} " % (
            "'" + str(i) + "'",
            j,
        )
        font_face = font_face + a
    start_html = "<html> <head> <style>"
    end_html = "</font></p> </body> </html>"
    mid_html = "</style> </head> <body> <p><font size=" + str(15) + ">"

    ranges = []
    ranges = list(range(65, 91))
    ranges += list(range(97, 123))
    emojis = [
        "😠",
        "😇",
        "🙅",
        "😶",
        "👋",
        "😄",
        "😒",
        "😵",
        "😼",
        "😰",
        "😒",
        "😡",
        "😂",
        "😫",
        "😖",
        "😱",
        "😲",
        "😽",
        "💗",
        "😢",
        "😐",
        "😳",
        "😈",
        "😜",
        "😁",
        "🙆",
        "💔",
        "😃",
        "😤",
        "😉",
        "💩",
        "👻",
        "❤",
        "♥",
        "🤗",
        "🧠",
        "🙏",
        "🌝",
        "🌚",
    ]
    digits = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    extra_list = list(string.punctuation) + emojis + digits  # add your characters here
    ranges += list(map(ord, extra_list))

    for i in ranges:
        if not os.path.exists(os.path.join(current_path, "../data/" + chr(i))):
            os.makedirs(os.path.join(current_path, "../data/" + chr(i)))
        # images = []

        for j in range(len(font_list)):
            img_name = os.path.join(
                current_path,
                "../data/" + chr(i) + "/" + font_list[j].split(".")[0] + ".png",
            )
            if not os.path.exists(img_name):
                print(chr(i))
                f = open("generate.html", "w")
                select_font = "* {font-family: %s}" % ("'" + str(j) + "'")
                text = (
                    start_html + font_face + select_font + mid_html + chr(i) + end_html
                )
                f.write(text)
                f.close()
                driver = webdriver.Chrome(os.path.join(current_path, "chromedriver"))
                driver.get("file://" + os.path.join(current_path, "generate.html"))
                driver.save_screenshot(img_name)
                img = save(img_name)
                driver.close()


if __name__ == "__main__":
    # text in the generate.html file
    generate_images()
