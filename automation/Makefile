SHELL=/bin/bash

newDay:
	cd /home/pnap32032/ &&\
	source ./env/bin/activate &&\
        cd automation/ &&\
	python3 data-clean.py

runNN:
	cd /home/pnap32032 &&\
	source ./env/bin/activate &&\
        cd automation/ &&\
	python3 neural-network.py

firstTime:
	cd .. &&\
	sudo apt-get install python3.6 &&\
	sudo apt-get install python3-venv &&\
	python3 -m venv env &&\
	source env/bin/activate &&\
	pip install --upgrade pip &&\
	python3 -m pip install -r requirements.txt &&\
	deactivate
	
