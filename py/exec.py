'''
launch env and solve via automated policy
'''
import socket
import struct
import subprocess
import time

import numpy as np

import policy

#PARAMETERS
H=32 # horizon length
SKIP=16 # window step
K=8 # similar examples count
ALT=True # use weighted-average based method instead of LP fitting to generate quadratic func

record=[] # stores timeseries data.
# format:
# target x y, actual x y, box x y theta
# goal:
# (400,300)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("127.0.0.1",65432))
    s.listen()

    # start env
    subprocess.Popen(["../pusht/Pusht.exe","--","--auto"])
    # load dataset
    dataset=policy.load_trajectories(H, SKIP)
    
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        buf=b''
        readout=[]
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


            if not len(readout) or len(data)>100:
                continue
            print(f"!!!!!!READOUT:{readout}")
            # using the last readout,
            # generate trajectory using policy
            traj=policy.policy_quadratic(np.array(readout[-5:]),dataset,K,"scs", alt=ALT)

            # execute half of the trajectory before replanning.
            for step in range(H//2):
                print(f"Sending {traj[step]}")
                msg=struct.pack("2q",int(traj[step][0]),int(traj[step][1]))
                conn.send(msg)
                time.sleep(0.1)

