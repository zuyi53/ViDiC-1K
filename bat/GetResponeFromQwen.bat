@echo off
call C:\ProgramData\miniconda3\Scripts\activate.bat C:\ProgramData\miniconda3
call conda activate ai

python "d:/GithubProject/ViDiC-1K/get_response_qwen3.py" ^
    --model_name qwen3-vl-8b-instruct ^
    --input_json_file "d:/GithubProject/ViDiC-1K/input_videos.json" ^
    --prompt_file "d:/GithubProject/ViDiC-1K/prompt/prompt_generate.txt" ^
    --batch_size 2 ^
    --fps 2.0 ^
    --gpu_memory_utilization 0.9