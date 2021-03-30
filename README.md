# Emoji Art Generator
This is a fun emoji art generator which I created because I was bored and wanted to relax for a while :P

Please try not to misuse this. 

## Usage
The current code supports images and text input. I'm open to new suggestions so please create an issue if needed.

### Command Line

The help section of the `emoji_art.py`:

```bash
$ python emoji_art.py -h
usage: emoji_art.py [-h] -mode MODE -input INPUT [-height HEIGHT] [-width WIDTH]
                    [-emoji1 EMOJI1] [-emoji2 EMOJI2] [-font_style FONT_STYLE]
                    [--square_crop] [--multiple_lines]

Generate an emoji art using text or image.

optional arguments:
  -h, --help            show this help message and exit
  -mode MODE            The mode for generation: either `image` or `text`.
  -input INPUT          The file path for `image` mode or the text for `text` mode. Text
                        mode can also include emojis if available in generated data.
  -height HEIGHT        The height of the image in 'character pixels' to be generated.
                        In `text` mode this is for each character.
  -width WIDTH          The width of the image in 'character pixels' to be generated. If
                        not provided, the height is equal to the width. In `text` mode
                        this is for each character.
  -emoji1 EMOJI1        The first emoji/string for the art.
  -emoji2 EMOJI2        The second emoji/string for the art.
  -font_style FONT_STYLE
                        The font-family to be used while generation. Note that with
                        cursive fonts, larger sizes are better.
  --square_crop         Square crop the image in case the image is rectangular. Only
                        used when mode is `image`. Cropping is done before generation.
  --multiple_lines      Generate the text in multiple lines with each character in a new
                        line. Only used when mode is `text`.

```
Currently, `Monospace`, `Courier`, 'AdreenaScript` and `StardustAdventure` font families are available. You can add you own and generate new fonts by running `generate.py`.
### Generating New Fonts and Characters
- In order to generate new fonts, you have to download the desired font's `.ttf` file and place it in the `generation/fonts` directory.

- In order to generate new characters, manually place them in `generation/generate.py` in the `extra_list`.

Please feel free to suggest new features, or report problems via issues. Have fun :)

### Output Examples

**Single Line Output**

```bash
$ python emoji_art.py -mode text -input "Hello Sir" -height 10 -font_style Monospace
🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🖤🤍🤍🤍🤍🤍🤍🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🤍🤍🤍🤍🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🖤🤍🤍🤍🤍🤍🤍🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🤍🤍🤍🤍🤍🤍🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🖤🤍🤍🤍🤍🤍🤍🖤🤍🤍🤍🤍🤍🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🤍🤍🤍🤍🤍
🤍🖤🤍🤍🤍🤍🤍🤍🖤🤍🤍🤍🤍🖤🤍🤍🤍🤍🖤🤍🤍🤍🤍🤍🤍🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🤍🤍🤍🤍🤍🤍🖤🤍🤍🤍🤍🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🖤🖤🤍🤍🤍🤍🤍🤍🤍🖤🖤🤍🤍🤍🤍
🤍🖤🤍🤍🤍🤍🤍🤍🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🤍🤍🤍🤍🤍🤍🤍🖤🖤🤍🤍🤍🤍🤍🤍🤍🖤🤍🤍🤍🤍🤍
🤍🖤🤍🤍🤍🤍🤍🤍🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🤍🤍🤍🤍🤍🤍🖤🖤🤍🤍🤍🤍🤍🤍🤍🖤🤍🤍🤍🤍🤍
🤍🖤🤍🤍🤍🤍🤍🤍🖤🤍🤍🤍🤍🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🤍🤍🤍🤍🤍🤍🖤🤍🤍🤍🤍🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🤍🤍🤍🤍🤍🤍🖤🤍🤍🤍🤍🤍🤍🖤🖤🤍🤍🤍🤍🤍🤍🤍🖤🤍🤍🤍🤍🤍
🤍🖤🤍🤍🤍🤍🤍🤍🖤🤍🤍🤍🤍🤍🖤🤍🤍🤍🖤🤍🤍🤍🤍🤍🤍🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🤍🤍🤍🤍🖤🤍🤍🤍🤍🤍🤍🤍🖤🖤🤍🤍🤍🤍🤍🤍🤍🖤🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍

```

**Muti-line Output**

```bash
$ python emoji_art.py -mode text -input "Hello Sir" -height 10 -font_style Monospace --multiple_lines
🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🖤🤍🤍🤍🤍🤍🤍🖤🤍
🤍🖤🤍🤍🤍🤍🤍🤍🖤🤍
🤍🖤🤍🤍🤍🤍🤍🤍🖤🤍
🤍🖤🖤🖤🖤🖤🖤🖤🖤🤍
🤍🖤🤍🤍🤍🤍🤍🤍🖤🤍
🤍🖤🤍🤍🤍🤍🤍🤍🖤🤍
🤍🖤🤍🤍🤍🤍🤍🤍🖤🤍
🤍🖤🤍🤍🤍🤍🤍🤍🖤🤍
🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍

🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🖤🤍🤍🖤🖤🤍
🤍🖤🤍🤍🤍🤍🖤🤍
🤍🖤🖤🖤🖤🖤🖤🤍
🤍🖤🤍🤍🤍🤍🤍🤍
🤍🖤🤍🤍🤍🤍🤍🤍
🤍🤍🖤🤍🤍🖤🖤🤍
🤍🤍🤍🤍🖤🤍🤍🤍

🤍🖤🖤🖤🖤🤍🤍🤍
🤍🤍🤍🖤🖤🤍🤍🤍
🤍🤍🤍🖤🖤🤍🤍🤍
🤍🤍🤍🖤🖤🤍🤍🤍
🤍🤍🤍🖤🖤🤍🤍🤍
🤍🤍🤍🖤🖤🤍🤍🤍
🤍🤍🤍🖤🖤🤍🤍🤍
🤍🖤🖤🖤🖤🖤🖤🤍

🤍🖤🖤🖤🖤🤍🤍🤍
🤍🤍🤍🖤🖤🤍🤍🤍
🤍🤍🤍🖤🖤🤍🤍🤍
🤍🤍🤍🖤🖤🤍🤍🤍
🤍🤍🤍🖤🖤🤍🤍🤍
🤍🤍🤍🖤🖤🤍🤍🤍
🤍🤍🤍🖤🖤🤍🤍🤍
🤍🖤🖤🖤🖤🖤🖤🤍

🤍🤍🤍🤍🤍🤍🤍🤍
🤍🖤🖤🤍🤍🖤🖤🤍
🤍🖤🤍🤍🤍🤍🖤🤍
🤍🖤🤍🤍🤍🤍🖤🤍
🤍🖤🤍🤍🤍🤍🖤🤍
🤍🖤🤍🤍🤍🤍🖤🤍
🤍🖤🖤🤍🤍🖤🖤🤍
🤍🤍🤍🤍🤍🤍🤍🤍






🤍🤍🤍🤍🖤🤍🤍🤍🤍🤍
🤍🖤🖤🤍🤍🤍🖤🖤🖤🤍
🤍🖤🤍🤍🤍🤍🤍🤍🖤🤍
🤍🖤🖤🤍🤍🤍🤍🤍🤍🤍
🤍🤍🖤🖤🖤🖤🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🖤🖤🤍🤍
🤍🤍🤍🤍🤍🤍🤍🤍🖤🤍
🤍🖤🤍🤍🤍🤍🤍🤍🖤🤍
🤍🖤🖤🖤🤍🤍🤍🖤🖤🤍
🤍🤍🤍🤍🖤🖤🤍🤍🤍🤍

🤍🤍🤍🖤🖤🤍🤍🤍
🤍🤍🤍🤍🤍🤍🤍🤍
🤍🖤🖤🖤🖤🤍🤍🤍
🤍🤍🤍🖤🖤🤍🤍🤍
🤍🤍🤍🖤🖤🤍🤍🤍
🤍🤍🤍🖤🖤🤍🤍🤍
🤍🤍🤍🖤🖤🤍🤍🤍
🤍🖤🖤🖤🖤🖤🖤🤍

🤍🤍🤍🤍🤍🖤🤍🤍
🤍🤍🖤🤍🖤🤍🤍🤍
🤍🤍🖤🖤🤍🤍🤍🤍
🤍🤍🖤🤍🤍🤍🤍🤍
🤍🤍🖤🤍🤍🤍🤍🤍
🤍🤍🖤🤍🤍🤍🤍🤍
🤍🤍🖤🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🤍🤍
```

**Image Example Output**

Original Image:
![monkey sticker](./sample_images/monkey.webp)

```bash
$ python emoji_art.py -mode image -input ./sample_images/monkey.webp -height 50
🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🤍🤍🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🤍🤍🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🖤🖤🖤🖤🖤🖤🤍🤍🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🖤🖤🖤🖤🖤🖤🖤🤍🤍🤍🤍🖤🖤🖤🖤🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🖤🖤🖤🖤🖤🖤🖤🤍🤍🖤🖤🖤🤍🖤🤍🤍🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🤍🖤🖤🖤🤍🖤🖤🤍🤍🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🤍🤍🤍🖤🖤🖤🖤🖤🤍🤍🤍🤍🤍🖤🖤🖤🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🤍🖤🖤🤍🖤🖤🖤🖤🤍🤍🖤🖤🤍🖤🖤🖤🤍🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🤍🤍🤍🖤🖤🤍🤍🖤🖤🖤🤍🤍🖤🖤🤍🖤🖤🖤🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🤍🖤🤍🤍🤍🤍🤍🖤🖤🖤🖤🖤🤍🖤🖤🖤🖤🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🤍🖤🖤🖤🖤🖤🖤🤍🤍🤍🤍🖤🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🤍🖤🖤🖤🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🖤🖤🖤🖤🤍🤍🤍🖤🖤🖤🖤🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🖤🖤🖤🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🤍🖤🖤🤍🖤🖤🖤🖤🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🖤🖤🖤🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🖤🖤🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🖤🤍🖤🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🖤🖤🤍🖤🖤🤍🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🖤🖤🖤🖤🤍🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🖤🖤🖤🖤🖤🖤🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🖤🖤🤍🤍🤍🤍🖤🖤🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🖤🤍🖤🤍🤍🖤🖤🖤🖤🤍🖤🖤🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🖤🖤🖤🖤🖤🤍🖤🖤🤍🖤🖤🖤🖤🖤🤍🤍🤍🤍🤍🤍🖤🖤🖤🖤🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🤍🖤🖤🖤🖤🤍🤍🤍🤍🖤🖤🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🤍🤍🤍🖤🖤🖤🤍🖤🤍🖤🤍🖤🤍🤍🤍
🤍🤍🤍🤍🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🖤🖤🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🖤🖤🖤🖤🤍🖤🤍🖤🖤🖤🖤🤍🤍🤍
🤍🤍🤍🤍🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🤍🖤🖤🖤🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🤍🤍🤍🖤🤍🤍🤍🖤🤍🖤🤍🤍
🤍🤍🤍🤍🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🤍🤍🤍🖤🖤🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🤍🤍🖤🤍🖤🖤🖤🤍🖤🖤🤍🤍
🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🖤🖤🖤🖤🤍🖤🤍🤍🖤🖤🖤🖤🖤🤍🤍🖤🖤🖤🖤🖤🖤🖤🖤🤍🤍🖤🖤🖤🖤🖤🖤🖤🤍🤍🤍
🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🖤🖤🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🤍🤍🤍🤍🤍🖤🤍🤍🤍🤍🖤🖤🖤🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🤍🖤🖤🖤🖤🤍🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🖤🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🖤🖤🖤🖤🖤🖤🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍🤍
```


