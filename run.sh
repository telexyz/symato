mkdir -p model && cd model
# [ -f 20B_tokenizer.json ] || wget https://raw.githubusercontent.com/BlinkDL/RWKV-LM/main/RWKV-v4neo/20B_tokenizer.json
# [ -f RWKV-4-Pile-169M-20220807-8023.pth ] || wget https://huggingface.co/BlinkDL/rwkv-4-pile-169m/resolve/main/RWKV-4-Pile-169M-20220807-8023.pth
[ -f symato-2816-vlc-23m.pth ] || wget https://huggingface.co/tiendung/symato-2816-vlc/resolve/main/symato-2816-vlc-23m.pth
cd ..
python3 model_run_f32.py
