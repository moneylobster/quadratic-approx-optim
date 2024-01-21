Requirements
--------------------------------------------------------------------------------
-Windows (on linux the pusht executable needs to be rebuilt using Godot)
-python libraries:
		-numpy
		-scipy
		-qpsolvers (pip install qpsolvers[scs]
		-faiss

Running
--------------------------------------------------------------------------------
	Training
The program needs data to run. This data already exists in the data/ folder, but
to add more data, run py/train.py. Using your mouse, push the blue box onto the
green square. Then press Esc to stop recording. On the python terminal, press Enter
to save the recording.

   Executing
Run py/exec.py. There are some parameters in the script that can be changed to alter
many aspects of the policy.

