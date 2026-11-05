# aurrium-os
Agentic AI powered OS using Termux Bootstrap.

Download the app on the release page. On the right side if you're on desktop.
https://github.com/sprites202/aurrium-os/releases/tag/aurrium-os-apk

The source code is also on releases due to its sheer size.
https://github.com/sprites202/aurrium-os/releases/tag/aurrium-os-source

Demo Video:
[![AurriumOS - Agentic AI Powered Virtual OS for Android](https://youtube.com)](https://youtube.com/shorts/9kMo26cutOo)

# Claimed vector database connection

The Termux bootstrap is capable of running our vector DB, but it currently requires hosting an external server for the embedding models. It is also yet to be parallelized by changing the structure to a contiguous page per cluster. To show its capability to run memory-mapped databases, we have included 2 files in the `mmaped_vector_db` folder:

- `store_signed_termux.cpp` – the memory-mapped vector store (builds for aarch64)
- `store_server_termux.py` – Flask test server (port 5020)

<img width="360" height="800" alt="image" src="https://github.com/user-attachments/assets/deb9caef-653a-40d0-ac88-6b492c9a51bb" />

GitHub folder: https://github.com/sprites202/aurrium-os/tree/main/mmaped_vector_db

## Run it in Termux

```bash
pkg install git clang python python-numpy
pip install flask

git clone https://github.com/sprites202/aurrium-os
cd aurrium-os/mmaped_vector_db

clang++ -O3 -std=c++17 -shared -fPIC -Wno-address-of-packed-member \
    -o libstore_signed.so store_signed_termux.cpp

nohup python store_server_termux.py > store_server.log 2>&1 &
```

## Test

```bash
curl http://localhost:5020/api/test_search
```

Sends a random vector and returns the top-k results with image paths. The first call seeds 2000 random records.

## Show an example image

```bash
mkdir -p uploaded_frames
cp /sdcard/<FOLDER>/<image>.jpg uploaded_frames/
```

Then open `http://localhost:5020/view` in the phone browser.

## Stop

```bash
pkill -f store_server_termux.py
```
