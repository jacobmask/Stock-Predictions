# Stock Prediction Application

## Installation
This installation will be used in a linux environment. You can do this through the student server, if you have an account connect to [yourname]@student.cs.uni.edu through an ssh shell.

Or this can be done having a virtualbox running ubuntu 18.04(preferred), a guide can be shown here:
https://brb.nci.nih.gov/seqtools/installUbuntu.html

This can still be done other ways, for our project we had a private virtual linux server, similar to the student server.  

### Cloning our Git repository
1. Make a directory on the linux server you are using from the command line: `mkdir application`
2. Get into that directory: `cd application`
3. Clone our repository: `git clone https://charon.cs.uni.edu/birkitaa/4620-predicting-stock-prices.git`
4. Get into that repository: `cd 4620-predicting-stock-prices`
You now have our project repository on your linux box.

### Run our project(Easy Way)
For our project we made a simple bash script that can setup everything for you to run our code. Or if you want to do it manually you can check our "Run our project(Hard way)".


### Run our project(Hard way)
Assuming you are in the `/application/4620-predicting-stock-prices` directory:
1. Create a virtual environment `python3 -m venv env`
2. Activate your environment `source env/bin/activate`
3. Make sure pip is up to date `pip install --upgrade pip`
4. 
