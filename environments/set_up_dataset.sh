#!/bin/bash

CURRENT_DIR=$(pwd)

if [[ ! "$CURRENT_DIR" == *"group-project-b3"* ]]; then
    echo "Error: Not in group-project-b3 directory"
    exit 1
fi

export HF_TOKEN=$(python environments/get_hf_token.py)

if [ ! -d ./data/MangaSegmentation ]; then
    hf download MS92/MangaSegmentation --repo-type=dataset --local-dir ./data/MangaSegmentation --token $HF_TOKEN
    rm -rf ./data/MangaSegmentation/.cache
else
    echo "./data/MangaSegmentation already exists, skipping download."
fi

if [[ ! -d ./data/MangaSegmentation/json_processed ]]; then
    gdown --fuzzy "https://drive.google.com/file/d/1IvwKFzQgSkDpJQT69wpHlChkhBtfm_p3/view?usp=sharing" -O ./data/MangaSegmentation/jsons_processed.zip
    unzip -o ./data/MangaSegmentation/jsons_processed.zip -d ./data/MangaSegmentation/
    rm ./data/MangaSegmentation/jsons_processed.zip
else
    echo "./data/MangaSegmentation/jsons_processed already exists, skipping download."
fi

if [ ! -d ./data/Manga109_released_2023_12_07 ]; then
    hf download hal-utokyo/Manga109 --repo-type dataset --include Manga109_released_2023_12_07.zip --local-dir ./data --token $HF_TOKEN
    unzip ./data/Manga109_released_2023_12_07.zip -d ./data
    rm ./data/Manga109_released_2023_12_07.zip
    rm -rf ./data/.cache
    rm -rf ./data/__MACOSX
else
    echo "./data/Manga109_released_2023_12_07 already exists, skipping download."
fi

# if [ ! -d ./data/Manga109s_released_2023_12_07 ]; then
#     hf download hal-utokyo/Manga109-s --repo-type dataset --include Manga109s_released_2023_12_07.zip --local-dir ./data --token $HF_TOKEN
#     unzip ./data/Manga109s_released_2023_12_07.zip -d ./data
#     rm ./data/Manga109s_released_2023_12_07.zip
#     rm -rf ./data/.cache
#     rm -rf ./data/__MACOSX
# else
#     echo "./data/Manga109s_released_2023_12_07 already exists, skipping download."
# fi

# if [ ! -d ./data/Human_Annotate_300 ]; then
#     gdown --fuzzy "https://drive.google.com/file/d/1nekCTGInk57Oe2cZvOjO7RRkUpS2mevw/view?usp=sharing" -O ./data/Human_Annotate_300.zip 
#     unzip -o ./data/Human_Annotate_300.zip -d ./data/
#     rm ./data/Human_Annotate_300.zip
# else
#     echo "./data/Human_Annotate_300 already exists, skipping download."
# fi
