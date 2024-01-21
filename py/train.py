'''
launch env and collect user data
'''
import socket
import struct
import subprocess
from datetime import datetime

import numpy as np

record=[] # stores timeseries data.
# format:
# target x y, actual x y, box x y theta
# goal:
# (400,300)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("127.0.0.1",65432))
    s.listen()

    # start env
    subprocess.Popen(["../pusht/Pusht.exe","--","--user"])
    
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        buf=b''
        while True:
            data = conn.recv(1024)
            if not data:
                break
            sp=data.split(b"start") # stuff to determine start of message
            buf+=sp[0]
            if len(sp)==2:
                if len(buf)==56:
                    readout=struct.unpack("@7q",buf) # parse data
                    record.append(readout) # append to record
                buf=sp[1]
            if len(sp)>2:
                buf=sp[-1]

print(f"Gathered {len(record)} items of data.")
recordnp=np.array(record)
print(f"Start:\n{recordnp[0]}\nEnd:\n{recordnp[-1]}")

if input("Save? (n for no)")!="n":
    np.save(f"data/{datetime.now().strftime('%d_%H%M%S')}.npy",recordnp)
    print("Saved.")
