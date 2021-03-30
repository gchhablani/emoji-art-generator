
import cv2
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

if __name__ == '__main__':

    parser = ArgumentParser(
        prog="emoji_art.py", description="Generate an emoji art using text or image."
    )
    parser.add_argument(
        "-mode", type=str, action="store", required=True, help="The mode for generation: either `image` or `text`."
    )
    parser.add_argument(
        "-input", type=str, action="store", required=True, help="The file path for `image` mode or the text for `text` mode. Text mode can also include emojis if available in generated data."
    )
    parser.add_argument(
        "-height", type=int, action="store", default=300, help="The height of the image in 'character pixels' to be generated. In `text` mode this is for each character."
    )
    parser.add_argument(
        "-width", type=int, action="store", default=None, required=False, help="The width of the image in 'character pixels' to be generated. If not provided, the height is equal to the width. In `text` mode this is for each character."
    )
    parser.add_argument(
        "-foreground_string", type=str, action="store", default='🖤', help="The foreground emoji/string for the art."
    )
    parser.add_argument(
        "-background_string", type=str, action="store", default='🤍', help="The background emoji/string for the art."
    )
    parser.add_argument(
        "-font_style", type=str, action="store", default="Monospace", help="The font-family to be used while generation. Note that with cursive fonts larger sizes are better."
    )
    parser.add_argument(
        "--square_crop", action="store_true", help="Square crop the image in case the image is rectangular. Only used when mode is `image`. Cropping is done before generation."
    )
    parser.add_argument(
        "--multiple_lines", action="store_true", help="Generate the text in multiple lines with each character in a new line. Only used when mode is `text`."
    )

    args = parser.parse_args()

    if args.width is None:
        args.width = args.height

    if args.mode == 'text':
        if args.multiple_lines:
            for char in args.input:
                if char == ' ':
                    print("\n"*4)
                    continue
                img = cv2.imread(f'data/{char}/{args.font_style}.png')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if char.isupper():
                    img = cv2.resize(
                        img, (args.height, args.width), interpolation=cv2.INTER_CUBIC)
                else:
                    img = cv2.resize(
                        img, (int(0.8*args.height), int(0.8*args.width)), interpolation=cv2.INTER_CUBIC)
                output = ''
                ret2, img = cv2.threshold(
                    img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # WhatsApp: '      ' # Linux: '  '
                new = np.where(img == 0, args.foreground_string,
                               args.background_string)
                for row in new:
                    output += ''.join(row)+'\n'
                print(output)
        else:
            output_arr = None
            for char in args.input:
                if char == ' ':
                    if output_arr is None:
                        output_arr = np.array(
                            [255, ]*(args.width)*args.height).reshape(args.height, -1)
                    else:
                        output_arr = np.concatenate((output_arr, np.array(
                            [255, ]*(args.width)*args.height).reshape(args.height, -1)), axis=1)
                    continue

                img = cv2.imread(f'data/{char}/{args.font_style}.png')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # print(char)
                # print(args.font_style)

                if char.isupper():
                    img = cv2.resize(
                        img, (args.height, args.width), interpolation=cv2.INTER_CUBIC)
                else:
                    img = cv2.resize(
                        img, (int(0.8*args.height), int(0.8*args.width)), interpolation=cv2.INTER_CUBIC)
                    # print(img.shape)
                    img = np.pad(
                        img, ((args.height-img.shape[0], 0), (args.width-img.shape[1], 0)), constant_values=((255, 255), (255, 255)))

                ret2, img = cv2.threshold(
                    img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if output_arr is None:
                    output_arr = img
                else:
                    output_arr = np.concatenate((output_arr, img), axis=1)
            output = ''
            # WhatsApp: '      ' # Linux: '  '
            new = np.where(output_arr == 0, args.foreground_string,
                           args.background_string)
            for row in new:
                output += ''.join(row)+'\n'
            print(output)

    elif args.mode == 'image':
        img = cv2.imread(args.input)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if args.square_crop:
            w, h = img.shape
            if w > h:
                img = img[w//2-h//2:w//2+h//2, :]
            else:
                img = img[:, h//2-w//2:h//2+w//2]

        img = cv2.resize(img, (args.width, args.height),
                         interpolation=cv2.INTER_CUBIC)
        ret2, img = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        new = np.where(img == 0, args.foreground_string,
                       args.background_string)
        output = ''
        for row in new:
            output += ''.join(row)+'\n'
        print(output)

    else:
        raise NotImplementedError(
            "Currently only `image` and `text` mode are supported.")
