import os

import pandas as pd
import streamlit as st

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))

CUSTOM_FONT = "JetBrains Mono"

CUSTOM_FONT_URL = """<link rel="preconnect" href="https://fonts.googleapis.com">
                     <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
                     <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono&display=block" rel="stylesheet">"""

THEMES = {
    "nord-light": {
        "base": "light",
        "primaryColor": "#5E81AC",
        "backgroundColor": "#ECEFF4",
        "secondaryBackgroundColor": "#E5E9F0",
        "textColor": "#2E3440",
    },
    "dracula": {
        "base": "dark",
        "primaryColor": "#BD93F9",
        "backgroundColor": "#282A36",
        "secondaryBackgroundColor": "#44475A",
        "textColor": "#F8F8F2",
    },
}

DEFAULT_THEME = THEMES["nord-light"]


def set_app_config(
    palette_name: object = DEFAULT_THEME,
    config_folder_path: str = os.path.join(ROOT_DIR, ".streamlit"),
    font_style: str = "monospace",
    max_upload_size: int = 1,
):

    palette = THEMES[palette_name]

    with open(os.path.join(config_folder_path, "config.toml"), "w") as f:
        f.write("[server]\n")
        f.write("maxUploadSize = {0}\n".format(max_upload_size))

        f.write("\n# theme: {0}\n".format(palette_name))
        f.write("[theme]\n")
        f.write("base = '{0}'\n".format(palette["base"]))

        f.write("primaryColor = '{0}'\n".format(palette["primaryColor"]))
        f.write("backgroundColor = '{0}'\n".format(palette["backgroundColor"]))
        f.write(
            "secondaryBackgroundColor = '{0}'\n".format(
                palette["secondaryBackgroundColor"]
            )
        )
        f.write("textColor = '{0}'\n".format(palette["textColor"]))

        f.write("\nfont = '{0}'".format(font_style))


def googlef_text(text: str, key="p", color=DEFAULT_THEME["textColor"], font=CUSTOM_FONT):

    text = """<{0} style="font-family: '{1}', monospace;color: {2};">{3}</{4}>""".format(
        key, font, color, text, key
    )

    st.markdown(text, unsafe_allow_html=True)


@st.cache(allow_output_mutation=True)
def read_csv(filename):
    df = pd.read_csv(filename, encoding="utf-8", names=["d (mm)", "??P", "Vb (cm??)"])
    return df
