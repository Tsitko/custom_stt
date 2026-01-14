from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="adilet11/CPT-Orpheus3B-ru",
    local_dir="/home/denis/.cache/orpheus/papacliff/orpheus-3b-0.1-ft-ru",          # куда скачать
    local_dir_use_symlinks=False       # чтобы лежали реальные файлы
)
