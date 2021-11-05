# Copyright (c) 2021, Hitachi America Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unicodedata


def _create_translation_table():
    table = {
        ord("\u200B"): "",  # Remove no-break spaces
        ord("\uFEFF"): "",  # Remove unicode BOM
        ord("\r"): " ",  # return to space
        ord("\n"): " "
    }
    # normalize unicodes
    table.update(dict([(ord(t), "\u301C") for t in "~\u007E\u02DC\u02F7\u0303\u0330\u0334\u223C\uFF5E\u301C"]))
    table.update(dict([(ord(t), "−") for t in "˗֊‐‑‒–⁃⁻₋−"]))
    table.update(dict([(ord(t), "ー") for t in "﹣－ｰ—―─━ー"]))
    return table


_TRANS_TABLE = _create_translation_table()


def preprocess_text(text: str):
    text = text.replace('\r\n', '\n')
    text = text.translate(_TRANS_TABLE)
    text = unicodedata.normalize('NFKC', text)
    text = text.replace('\t', '  ')
    # remove all hidden characters such as control characters
    text = ''.join(c for c in text if c.isprintable())
    return text.rstrip()
