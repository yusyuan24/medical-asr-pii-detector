# AICUP2025_TEAM_7464

本專案為 AICUP 2025 醫病語音個資辨識競賽提交程式碼，實作中英文語音轉文字（ASR）與語音個資命名實體辨識（PHI NER）任務  

## 環境配置與安裝
Windows11  
Python 3  
CUDA 11.8  
cuDNN 8.6 

主要使用程式語言為 Python 3.11，主要套件與函式庫包括：torchaudio 2.1.0、transformers 4.5.0 以及 librosa 等。


##  模型與資料來源
英文語音識別採用 Hugging Face 提供的 Whisper 模型（openai/whisper-small）作為語音轉文字（ASR）之核心工具，模型來源：https://huggingface.co/openai/whisper-small    
對中中文語音分，，採採用模模大的的 Whisper 模型（openai/whisper-large），以提升中文語音辨識的準確率 ， 模型來源：https://huggingface.co/openai/whisper-large  

語音敏感個人資料辨識採使用 deepseek-ai/deepseek-llm-7b-base 模型 作為命名實體識別與語意理解的基礎模型，模型來源：https://huggingface.co/deepseek-ai/deepseek-llm-7b-base  

模型訓練資料來源取自 2025 AICUP 醫病語音敏感個人資料辨識競賽主辦方

## 專案結構簡述
```
├── AICUP2025_TEAM_7464.ipynb   # 主程式（Jupyter Notebook）  
├── AICUP.py                    # 主程式中使用到的自訂函式（功能模組）  
├── Training_Dataset.zip        # 音檔與標註資料（不公開）  
└── README.md                   # 專案說明文件  
```