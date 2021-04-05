SHELL=/bin/bash

newDay:
	source ./test1/bin/activate &&\
	python3 data-clean.py

runRNN:
	source ./test1/bin/activate &&\
	python3 RNN.py
