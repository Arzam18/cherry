#!/usr/bin/env python3

import urllib.request
import hashlib
import os

def main():
    name = "kiwi-v3"
    hash = "83b0c14fa69abd7a6e473290abb40a0c5315bb1fa6976d3e618e7743f6219312"
    path = "./networks/default.nnue"

    os.makedirs(os.path.dirname(path), exist_ok=True)

    try:
        if hashlib.sha256(open(path, "rb").read()).digest().hex() == hash:
            print("Net already exists!")
            return
    except OSError:
        pass

    print(f"Downloading net {name} to {path}")
    net = urllib.request.urlopen(
        f"https://github.com/Teccii/cherry-networks/releases/download/{name}/{name}.nnue"
    ).read()
    if hashlib.sha256(net).digest().hex() != hash:
        print("Invalid hash!")
        exit(1)
    open(path, "wb").write(net)

if __name__ == "__main__":
    main()
