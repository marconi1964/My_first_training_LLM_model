#!/bin/bash

# make sure set up the pyenv
# pyenv virtualenv pyenv 3.12.0 llm_env
# pyenv activate llm_env
# source ~/.bashrc

# chmod +x setup.sh

# 安裝必要套件
# brew install python
brew install xz
LDFLAGS="-L$(brew --prefix xz)/lib" CPPFLAGS="-I$(brew --prefix xz)/include" pyenv install 3.12.0
pip install torch transformers datasets accelerate

# 下載 WikiText-103 資料集（Hugging Face 直接加載，無需手動下載）
# 此步驟會在 Python 程式碼中自動處理
