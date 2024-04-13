This is a shrunk polygon-based scene text detector and real-time text detector. This repository is built on PAN, FAST. We tested the code in the environment of Arch Ubuntu20.04+Python3.8.
1.	sh ./complie.sh
2.	cd models/backbone/assets/dcn
3.	sh Makefile.sh
4.	for training: python train.py config/ctw640.py
5.	for testing: cd eval sh eval_ctw.sh python test.py config/ctw640.py [your checkpoints] --ema
6.	for speed: python test.py config/ctw640.py --report-speed
7.	for visulization: python visualize.py --dataset ctw --show-gt
