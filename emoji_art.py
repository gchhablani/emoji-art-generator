import math
import os
import string
import time
from argparse import ArgumentParser
from sys import int_info
from types import new_class

import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from selenium import webdriver

#
# https: // stackoverflow.com/questions/41721734/take-screenshot-of-full-page-with-selenium-python-with-chromedriver
from selenium.webdriver.chrome.options import Options
from skimage.segmentation import slic
from tqdm.auto import tqdm

# def test_fullpage_screenshot(self):
#     chrome_options = Options()
#     chrome_options.add_argument('--headless')
#     chrome_options.add_argument('--start-maximized')
#     driver = webdriver.Chrome(chrome_options=chrome_options)
#     driver.get("yoururlxxx")
#     time.sleep(2)

#     # the element with longest height on page
#     ele = driver.find_element(
#         "xpath", '//div[@class="react-grid-layout layout"]')
#     total_height = ele.size["height"]+1000

#     driver.set_window_size(1920, total_height)  # the trick
#     time.sleep(2)
#     driver.save_screenshot("screenshot1.png")
#     driver.quit()


def generate_html_around_str(strin):
    font_face = ""
    a = "@font-face { font-family: '0'; src: url('./generation/fonts/Monospace.ttf')} "

    font_face = font_face + a
    start_html = "<!DOCTYPE html>\n<html> <head> <meta charset=“UTF-8”> <style>"
    end_html = "</font></p> </body> </html>"
    mid_html = (
        "</style> </head> <body> <p style='background-color:#FAFAFA;display: inline-block;'><font size="
        + str(5)
        + ">"
    )

    f = open("out.html", "w")
    select_font = "* {font-family: '0'}"
    text = start_html + font_face + select_font + mid_html + strin + end_html
    f.write(text)
    f.close()


def save(path, strip_scrollbar=True):
    a = cv2.imread(path, 0)
    if strip_scrollbar:
        a = a[:, :-20]
    a[a > 250] = 255
    a[a <= 250] = 0

    w = np.argmin(a, axis=1)
    w[w == 0] = 5000
    left = min(w)

    h = np.argmin(a, axis=0)
    h[h == 0] = 5000
    top = min(h)

    a = np.rot90(a)
    h = np.argmin(a, axis=0)
    h[h == 0] = 5000
    right = min(h)
    right = a.shape[0] - right

    a = np.rot90(a)
    w = np.argmin(a, axis=0)
    w[w == 0] = 5000
    bottom = min(w)
    bottom = a.shape[0] - bottom

    a = cv2.imread(path)
    a = a[top - 1: bottom + 1, left - 1: right + 1]
    a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    plt.imsave("out.pdf", a)
    return a


html_path = "out.html"
current_path = "/".join(os.path.abspath(__file__).split("/")[:-1])


def fix_html_emojis(html):
    new_out = []
    bytes = html.split(";")
    for byte in bytes:
        if new_out == []:
            new_out.append(byte)
        else:
            if new_out[-1] != byte:
                new_out.append(byte)
            else:
                print(byte)
                print(new_out)
    return ";".join(new_out)


def open_html_in_browser_and_save(html_path):

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--start-maximized")
    driver = webdriver.Chrome(
        "./generation/chromedriver", options=chrome_options)
    driver.get("file://" + os.path.join(current_path, html_path))
    driver.execute_script("document.body.style.zoom='25%'")
    # driver.fullscreen_window()
    time.sleep(1)

    ele = driver.find_element("xpath", "//p")
    total_width = ele.size["width"]
    total_height = ele.size["height"]

    driver.set_window_size(total_width, total_height)  # the trick
    time.sleep(2)
    driver.save_screenshot("out.png")
    a = save("out.png")


def get_super_pixels(image, n_segments):
    label = slic(image, n_segments=n_segments)
    im_rp = image.reshape((image.shape[0] * image.shape[1], image.shape[2]))
    sli_1d = np.reshape(label, -1)
    uni = np.unique(sli_1d)
    uu = np.zeros(im_rp.shape)
    for i in uni:
        loc = np.where(sli_1d == i)[0]
        # print(loc)
        mm = np.mean(im_rp[loc, :], axis=0)
        uu[loc, :] = mm
    oo = np.reshape(uu, [image.shape[0], image.shape[1], image.shape[2]]).astype(
        "uint8"
    )

    return oo


with open("generation/emojis.json") as f:
    emojis = json.load(f)

choose_list_with_html = {emoji["emoji"]: fix_html_emojis(emoji["html"])
                         for emoji in emojis["emojis"] if ("flag" not in emoji["category"] and "geometric" not in emoji["category"])}

choose_list = list(choose_list_with_html.keys())
ignore_emoji_list = [
    "▪",
    "▪️",
    "♦",
    "▫️",
    "♠️",
    "®️",
    "🔹",
    "↕️",
    "↔",
    "☺",
    "◽",
    "♣️",
    "♀",
    "♥",
    "♀️",
    "↕",
    "☀",
    "♎",
    "⚒",
    "⬇️",
    "✖",
    "⬆️",
    "♍",
    "⚱",
    "☂",
    "⁉️",
    "✝",
    "‼",
    "❣️",
    "☔",
    "↘️",
    "↩",
    "⚔",
    "◻️",
    "✴️",
    "✒️",
    "✌",
    "♟",
    "❣",
    "☘️",
    "✉️",
    "✈",
    "✳",
    "⚓",
    "❤️",
    "☁",
    "♻️",
    "✂",
    "™️",
    "✈️",
    "♥️",
    "♣",
    "✉",
    "✝️",
    "☦",
    "⚙️",
    "®",
    "♏",
    "☘",
    "™",
    "♦️",
    "⚗",
    "↙️",
    "©️",
    "⚕",
    "♊",
    "❄",
    "☹️",
    "☮️",
    "♈",
    "⌨",
    "☃️",
    "⚛️",
    "♾️",
    "☑",
    "⚒️",
    "⬅️",
    "♐",
    "⬆",
    "☃",
    "✡",
    "♨",
    "♓",
    "⬇",
    "↘",
    "☯️",
    "☑️",
    "♨️",
    "☸",
    "◼",
    "⏏️",
    "⚰️",
    "♑",
    "♂️",
    "©",
    "◀️",
    "♻",
    "☦️",
    "‼️",
    "☺️",
    "✌️",
    "✴",
    "⚕️",
    "❇️",
    "↔️",
    "▫",
    "✒",
    "#",
    "◾",
    "✍",
    "♟️",
    "☎️",
    "⏏",
    "☂️",
    "⚖️",
    "♠",
    "🏿",
    "🏾",
    "🏽",
    "🏼",
    "🏻",

]

# chrome_ignore_emoji_list = ['▪', '▪️', '♦', '▫️', '♠️', '®️', '🔹', '↕️', '↔', '☺', '◽', '♣️', '♀', '♥', '♀️', '↕', '‼', '◻️',
#                      '™️', '♥️', '♣', '®', '™', '♦️', '©️', '♂️', '©', '‼️', '☺️', '↔️', '▫', '#', '◾', '♠', '🏿', '🏾', '🏽', '🏼', '🏻',
#                      ]


emojis = {
    " o/": " 👋",
    " </3": " 💔",
    " <3": " ❤",
    " 8-D": " 😁",
    " 8D": " 😁",
    " :-D": " 😁",
    " =-3": " 😁",
    " =-D": " 😁",
    " =3": " 😁",
    " =D": " 😁",
    " B^D": " 😁",
    " X-D": " 😁",
    " XD": " 😁",
    " x-D": " 😁",
    " xD": " 😁",
    " :')": " 😂",
    " :'-)": " 😂",
    " :-))": " 😃",
    " 8)": " 😄",
    " :)": " 😄",
    " :-)": " 😄",
    " :3": " 😄",
    " :D": " 😄",
    " :]": " 😄",
    " :^)": " 😄",
    " :c)": " 😄",
    " :o)": " 😄",
    " :}": " 😄",
    " :っ)": " 😄",
    " =)": " 😄",
    " =]": " 😄",
    " 0:)": " 😇",
    " 0:-)": " 😇",
    " 0:-3": " 😇",
    " 0:3": " 😇",
    " 0;^)": " 😇",
    " O:-)": " 😇",
    " 3:)": " 😈",
    " 3:-)": " 😈",
    " }:)": " 😈",
    " }:-)": " 😈",
    " *)": " 😉",
    " *-)": " 😉",
    " :-,": " 😉",
    " ;)": " 😉",
    " ;-)": " 😉",
    " ;-]": " 😉",
    " ;D": " 😉",
    " ;]": " 😉",
    " ;^)": " 😉",
    " :-|": " 😐",
    " :|": " 😐",
    " :(": " 😒",
    " :-(": " 😒",
    " :-<": " 😒",
    " :-[": " 😒",
    " :-c": " 😒",
    " :<": " 😒",
    " :[": " 😒",
    " :c": " 😒",
    " :{": " 😒",
    " :っC": "😒",
    " %)": " 😖",
    " %-)": " 😖",
    " :-P": " 😜",
    " :-b": " 😜",
    " :-p": " 😜",
    " :-Þ": " 😜",
    " :-þ": " 😜",
    " :P": " 😜",
    " :b": " 😜",
    " :p": " 😜",
    " :Þ": " 😜",
    " :þ": " 😜",
    " ;(": " 😜",
    " =p": " 😜",
    " X-P": " 😜",
    " XP": " 😜",
    " d:": " 😜",
    " x-p": " 😜",
    " xp": " 😜",
    " :-||": " 😠",
    " :@": " 😠",
    " :-.": " 😡",
    " :-/": " 😡",
    " :/": " 😡",
    " :L": " 😡",
    " :S": " 😡",
    " :\\": " 😡",
    " =/": " 😡",
    " =L": " 😡",
    " =\\": " 😡",
    " :'(": " 😢",
    " :'-(": " 😢",
    " ^5": " 😤",
    " ^<_<": " 😤",
    " o/\\o": " 😤",
    " |-O": " 😫",
    " |;-)": " 😫",
    " :###..": " 😰",
    " :-###..": " 😰",
    " D-':": " 😱",
    " D8": " 😱",
    " D:": " 😱",
    " D:<": " 😱",
    " D;": " 😱",
    " D=": " 😱",
    " DX": " 😱",
    " v.v": " 😱",
    " 8-0": " 😲",
    " :-O": " 😲",
    " :-o": " 😲",
    " :O": " 😲",
    " :o": " 😲",
    " O-O": " 😲",
    " O_O": " 😲",
    " O_o": " 😲",
    " o-o": " 😲",
    " o_O": " 😲",
    " o_o": " 😲",
    " :$": " 😳",
    " #-)": " 😵",
    " :#": " 😶",
    " :&": " 😶",
    " :-#": " 😶",
    " :-&": " 😶",
    " :-X": " 😶",
    " :X": " 😶",
    " :-J": " 😼",
    " :*": " 😽",
    " :^*": " 😽",
    " ಠ_ಠ": " 🙅",
    " *\\0/*": " 🙆",
    " \\o/": " 🙆",
    " :>": " 😄",
    " >.<": " 😡",
    " >:(": " 😠",
    " >:)": " 😈",
    " >:-)": " 😈",
    " >:/": " 😡",
    " >:O": " 😲",
    " >:P": " 😜",
    " >:[": " 😒",
    " >:\\": " 😡",
    " >;)": " 😈",
    " >_>^": " 😤",
}


def is_emoji(s):
    range_min = ord("\U0001F300")  # 127744
    range_max = ord("\U0001FAD6")  # 129750
    range_min_2 = 126980
    range_max_2 = 127569
    range_min_3 = 169
    range_max_3 = 174
    range_min_4 = 8205
    range_max_4 = 12953

    char_code = ord(s)
    if range_min <= char_code <= range_max:
        # or range_min_2 <= char_code <= range_max_2 or range_min_3 <= char_code <= range_max_3 or range_min_4 <= char_code <= range_max_4:
        return True
    elif range_min_2 <= char_code <= range_max_2:
        return True
    elif range_min_3 <= char_code <= range_max_3:
        return True
    elif range_min_4 <= char_code <= range_max_4:
        return True
    return False


scale = {
    "upper": 1,
    "digit": 1,
    "emoji": 0.9,
    "regular_punct": 0.9,
    "small_punct": 0.6,
    "other": 0.9,
}


def get_char_type(char):
    if char.isupper():
        return "upper"
    elif char.isdigit():
        return "digit"
    elif is_emoji(char):
        return "emoji"
    elif char in string.punctuation:
        if char not in ",.;":
            return "regular_punct"
        else:
            return "small_punct"
    return "other"


def generate_char(char, args):
    img = cv2.imread(f"data/{char}/{args.font_style}.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    typ = get_char_type(char)
    scaling = scale[typ]
    img = cv2.resize(
        img,
        (math.floor(scaling * args.width), math.floor(scaling * args.height)),
        interpolation=cv2.INTER_CUBIC,
    )
    if scaling != 1:
        try:
            img = np.pad(
                img,
                ((args.height - img.shape[0], 0),
                 (0, args.width - img.shape[1])),
                constant_values=((255, 255), (255, 255)),
            )
        except Exception as e:
            print(
                f"Unable to pad with image shape: {img.shape}, args width: {args.width}, and args height: {args.height}."
            )
            print(f"Exception Occurred: {e}")
    if typ == "emoji":
        if args.emoji_thresh is not None and args.emoji_thresh >= 0:
            img[img >= args.hard_thresh] = 255
            img[img < args.hard_thresh] = 0
        else:
            ret2, img = cv2.threshold(
                img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        ret2, img = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if args.no_output:
        plt.imshow(img)
        plt.show()

    return img


# WhatsApp Space Equivalent: '      ' # Linux Space Equivalent: '  '


if __name__ == "__main__":

    parser = ArgumentParser(
        prog="emoji_art.py", description="Generate an emoji art using text or image."
    )
    parser.add_argument(
        "-mode",
        type=str,
        action="store",
        required=True,
        help="The mode for generation: either `image` or `text`.",
    )
    parser.add_argument(
        "-input",
        type=str,
        action="store",
        required=True,
        help="The file path for `image` mode or the text for `text` mode. Text mode can also include emojis if available in generated data.",
    )
    parser.add_argument(
        "-height",
        type=int,
        action="store",
        default=None,
        required=False,
        help="The height of the image in 'character pixels' to be generated. If not provided, the height is equal to the width in `text` mode, and scaled according to width in `image` mode. If both not provided, width of 300 is used with respective height. In `text` mode this is for each character.",
    )
    parser.add_argument(
        "-width",
        type=int,
        action="store",
        default=None,
        required=False,
        help="The width of the image in 'character pixels' to be generated. If not provided, the width is equal to the height in `text` mode, and scaled according to height in `image` mode. If both not provided, width of 300 is used with respective height. In `text` mode this is for each character.",
    )
    parser.add_argument(
        "-foreground_string",
        type=str,
        action="store",
        default="🖤",
        help="The foreground emoji/string for the art.",
    )
    parser.add_argument(
        "-background_string",
        type=str,
        action="store",
        default="🤍",
        help="The background emoji/string for the art.",
    )
    parser.add_argument(
        "-font_style",
        type=str,
        action="store",
        default="Monospace",
        help="The font-family to be used while generation. Note that with cursive fonts larger sizes are better.",
    )
    parser.add_argument(
        "-align_char",
        type=str,
        action="store",
        default="",
        help="Generate the rows with a character in the front to align them properly. Useful for sending over messenger apps which strip the initial space. Only used in `text` mode.",
    )

    parser.add_argument(
        "-emoji_thresh",
        type=int,
        action="store",
        default=-1,
        help="The emoji threshold used for emojis. Only used in `text` mode. If negative, then Otsu binarization is used.",
    )
    parser.add_argument(
        "-n_segments",
        type=int,
        action="store",
        default=0,
        help="The number of segments (approx) to be created if using super-pixels. Only used with `image` mode in `auto_color` setting.",
    )
    parser.add_argument(
        "--auto_color",
        action="store_true",
        help="If the output is to be colored using emojis automatically. Only used with `image` mode.",
    )
    parser.add_argument(
        "--fill_random",
        action="store_true",
        help="If the output is to be colored randomly using emojis in `auto_color` mode.",
    )
    parser.add_argument(
        "--square_crop",
        action="store_true",
        help="Square crop the image in case the image is rectangular. Only used when mode is `image`. Cropping is done before generation.",
    )
    parser.add_argument(
        "--multiple_lines",
        action="store_true",
        help="Generate the text in multiple lines with each character in a new line. Only used when mode is `text`.",
    )

    parser.add_argument(
        "--no_output",
        action="store_true",
        help="If the output is not to be printed, and only binarized image is to be shown. This is only useful for testing.",
    )

    args = parser.parse_args()
    new_line_str = "<br>"

    if args.width is None and args.height is None:
        args.width = 300

    if args.mode == "text":
        if args.width is None:
            args.width = args.height
        elif args.height is None:
            args.height = args.width

        for key in emojis:
            if key in args.input:
                args.input = args.input.replace(key, emojis[key])
        if args.multiple_lines:
            for char in args.input:
                if char == " ":
                    print(new_line_str * int(args.height))
                    continue
                img = generate_char(char, args)

                new = np.where(img == 0, args.foreground_string,
                               args.background_string)

                if not args.no_output:
                    output = ""
                    for row in new:
                        output += args.align_char + "".join(row) + new_line_str
                    print(output)
        else:
            output_arr = None
            for char in args.input:
                if char == " ":
                    if output_arr is None:
                        output_arr = np.array(
                            [
                                255,
                            ]
                            * (args.width)
                            * args.height
                        ).reshape(args.height, -1)
                    else:
                        output_arr = np.concatenate(
                            (
                                output_arr,
                                np.array(
                                    [
                                        255,
                                    ]
                                    * (args.width)
                                    * args.height
                                ).reshape(args.height, -1),
                            ),
                            axis=1,
                        )
                    continue

                img = generate_char(char, args)

                if output_arr is None:
                    output_arr = img
                else:
                    output_arr = np.concatenate((output_arr, img), axis=1)
            # WhatsApp: '      ' # Linux: '  '
            new = np.where(
                output_arr == 0, args.foreground_string, args.background_string
            )
            if not args.no_output:
                output = ""
                for row in new:
                    output += args.align_char + "".join(row) + new_line_str
                print(output)

    elif args.mode == "image":
        img = cv2.imread(args.input)
        if not args.auto_color and not args.n_segments > 0:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if args.width is None:
                args.width = int(args.height * img.shape[1] / img.shape[0])
            elif args.height is None:
                args.height = int(args.width * img.shape[0] / img.shape[1])
            if args.square_crop:
                w, h = img.shape
                if w > h:
                    img = img[w // 2 - h // 2: w // 2 + h // 2, :]
                else:
                    img = img[:, h // 2 - w // 2: h // 2 + w // 2]
            img = cv2.resize(
                img, (args.width, args.height), interpolation=cv2.INTER_CUBIC
            )
            ret2, img = cv2.threshold(
                img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            new = np.where(img == 0, args.foreground_string,
                           args.background_string)
            if not args.no_output:
                output = ""
                for row in new:
                    output += "".join(row) + new_line_str
                print(output)
            else:
                plt.imshow(img)
                plt.show()
        elif not args.auto_color and args.n_segments > 0:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = get_super_pixels(img, args.n_segments)
            plt.imsave("superpixels.png", img, dpi=10000)
        else:
            if args.width is None:
                args.width = int(args.height * img.shape[1] / img.shape[0])
            elif args.height is None:
                args.height = int(args.width * img.shape[0] / img.shape[1])
            if args.square_crop:
                # print(img.shape)
                w, h, _ = img.shape
                if w > h:
                    img = img[w // 2 - h // 2: w // 2 + h // 2, :]
                else:
                    img = img[:, h // 2 - w // 2: h // 2 + w // 2]
            img = cv2.resize(
                img, (args.width, args.height), interpolation=cv2.INTER_CUBIC
            )
            if args.n_segments > 0:
                img = get_super_pixels(img, args.n_segments)
            # https: // matplotlib.org/matplotblog/posts/emoji-mosaic-art/

            if not args.no_output:

                emoji_mean_df = pd.read_csv(
                    "generation/emojis_to_mean.tsv", sep="\t", header=None
                )
                emoji_mean_df = emoji_mean_df[~emoji_mean_df[0].isin(
                    ignore_emoji_list)]
                emoji_mean_df = emoji_mean_df[emoji_mean_df[0].isin(
                    choose_list)]
                print(emoji_mean_df.shape)
                emoji_mean_df[1] = emoji_mean_df[1].apply(eval)
                # Fix for white
                tree = KDTree(
                    np.array(list(emoji_mean_df[1]))-np.array([15, 15, 15]))
                shape = img.shape
                img = img.reshape(-1, 3)

                new = np.array([])
                for pixel in tqdm(img):
                    _, index = tree.query(pixel)
                    new = np.append(
                        new, choose_list_with_html[emoji_mean_df[0].iloc[index]])
                new = new.reshape(shape[:-1])
                output = ""
                for row in new:
                    output += "".join(row) + new_line_str
                generate_html_around_str(output)
                open_html_in_browser_and_save(html_path)
                # print(output)
            else:
                plt.imshow(img)
                plt.show()

    else:
        raise NotImplementedError(
            "Currently only `image` and `text` mode are supported."
        )

        # OLD HSV Alternative
        # for key in color_to_hsv:
        #     color_range = color_to_hsv[key]
        #     mask = cv2.inRange(
        #         img, np.array(color_range[1]), np.array(
        #             color_range[0])
        #     )
        #     if not args.fill_random:
        #         if key in color_to_emoji:
        #             new = np.where(mask, color_to_emoji[key], new)
        #         else:
        #             new = np.where(mask, "  ", new)
        #     else:
        #         lis = list(color_to_emoji.keys())
        #         if key in color_to_emoji:
        #             new = np.where(
        #                 mask, color_to_emoji[lis[np.random.randint(0, len(lis))]], new)
        #         else:
        #             new = np.where(mask, "  ", new)
