SHELL=/bin/bash

newDay:
	cd .. &&\
	source ./env/bin/activate &&\
        cd automation/
	python3 data-clean.py

runRNN:
	cd .. &&\
	source ./env/bin/activate &&\
        cd automation/
	python3 neural-network.py

setupEnv:
	cd .. &&\
	python3 -m venv env
	
	
