import json
import os
from io import BytesIO  # python 3
from sys import getsizeof

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from selenium import webdriver
from tqdm.auto import tqdm
from wcwidth import wcswidth


def remove_color(img, rgba):
    data = np.array(img.convert("RGBA"))  # rgba array from image
    pixels = data.view(dtype=np.uint32)[..., 0]  # pixels as rgba uint32
    data[..., 3] = np.where(
        pixels == np.uint32(rgba), np.uint8(0), np.uint8(255)
    )  # set alpha channel
    return Image.fromarray(data)


# https://gist.github.com/oliveratgithub/0bf11a9aff0d6da7b46f1490f86a71eb/
with open("emojis.json") as f:
    emojis = json.load(f)

emojis = [emoji["emoji"] for emoji in emojis["emojis"]]
# plt.ion()
np.random.shuffle(emojis)


def get_mean(path):
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

    img = cv2.imread(path)

    # # plt.imshow(img[top - 1: bottom + 1, left - 1: right + 1])
    # # plt.show()
    # # plt.close('all')
    # mask = (img[:, :, 0] == 255) & (
    #     img[:, :, 1] == 255) & (img[:, :, 0] == 255)

    # mask = np.expand_dims(mask, axis=2)
    # mask_3d = np.concatenate([mask, mask, mask], axis=2)
    # int_mask_3d = 1 - mask_3d.astype(np.uint8)

    # # Mean of only non-white pixelss
    # # plt.imshow(int_mask_3d[top - 1: bottom + 1, left - 1: right + 1]
    # #            * img[top - 1: bottom + 1, left - 1: right + 1],
    # #            )
    # # plt.show()
    return list(
        np.mean(
            img[top - 1: bottom + 1, left - 1: right + 1],
            axis=(0, 1),
        )
    )  # BGR mean #include extra white which is actually useful


def generate_images():
    means = []
    current_path = "/".join(os.path.abspath(__file__).split("/")[:-1])
    font_list = ["Monospace.ttf"]
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

    df = None
    if os.path.exists("emojis_to_mean.tsv"):
        df = pd.read_csv("emojis_to_mean.tsv", sep="\t", header=None)
    if df is not None:
        for emoji in df[0].values:
            try:
                emojis.remove(emoji)
            except:
                print(emoji)
                raise ValueError()
    # ranges = list(map(ord, emojis))
    ranges = emojis
    for i in tqdm(ranges):
        # print(wcswidth(i))
        # inp = ""
        # # inp = input(
        # #     f"Press Enter to add this emoji: {i}. Press any other key to skip.")
        # if inp == "":
        for j in range(len(font_list)):
            f = open("generate.html", "w")
            select_font = "* {font-family: %s}" % ("'" + str(j) + "'")
            text = start_html + font_face + select_font + mid_html + i + end_html
            f.write(text)
            f.close()
            driver = webdriver.Chrome(
                os.path.join(current_path, "chromedriver"))
            driver.get("file://" + os.path.join(current_path, "generate.html"))
            # with Image.open(BytesIO(driver.get_screenshot_as_png())) as img:
            #     with remove_color(img, 0xffffffff) as img2:
            #         img2.save(r"temp.png")
            driver.save_screenshot("temp.png")
            mean = get_mean("temp.png")
            with open("emojis_to_mean.tsv", "a") as f:
                f.write(f"{i}\t{str(mean)}\n")
            driver.close()
            means.append(mean)

    return means


if __name__ == "__main__":
    means = generate_images()
    # with open("emojis_to_mean.tsv", "a") as f:
    #     for i in range(len(emojis)):
    #         if i != len(emojis) - 1:
    #             f.write(f"{emojis[i]}\t{str(means[i])}\n")
    #         else:
    #             f.write(f"{emojis[i]}\t{str(means[i])}")
