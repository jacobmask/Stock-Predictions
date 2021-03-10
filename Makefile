SHELL=/bin/bash

help:
	@echo Choose \(newday, train\)

newday:
	python3 pull.py


train:
	python3 RNN.py
