from sys import int_info
import cv2
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import string
import math


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
    "other": 0.9
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
    img = cv2.resize(img, (math.floor(scaling*args.width), math.floor(scaling*args.height)),
                     interpolation=cv2.INTER_CUBIC)
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
                f"Unable to pad with image shape: {img.shape}, args width: {args.width}, and args height: {args.height}.")
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
                    print("\n" * int(args.height))
                    continue
                img = generate_char(char, args)

                new = np.where(img == 0, args.foreground_string,
                               args.background_string)

                if not args.no_output:
                    output = ""
                    for row in new:
                        output += args.align_char + "".join(row) + "\n"
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
                    output += args.align_char + "".join(row) + "\n"
                print(output)

    elif args.mode == "image":
        img = cv2.imread(args.input)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if args.width is None:
            args.width = int(args.height*img.shape[1]/img.shape[0])
        elif args.height is None:
            args.height = int(args.width*img.shape[0]/img.shape[1])
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
                output += "".join(row) + "\n"
            print(output)
        else:
            plt.imshow(img)
            plt.show()

    else:
        raise NotImplementedError(
            "Currently only `image` and `text` mode are supported."
        )
