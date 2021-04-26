# Stock Prediction Application

## First time Installation and setup guide
This can be done having a virtualbox running ubuntu 18.04(preferred), a guide can be shown here:
https://brb.nci.nih.gov/seqtools/installUbuntu.html

This can still be done other ways, for our project we had a private virtual linux server. The student server cannot work our project without changes to some files, since the student server is on python3.5 and we have it on 3.6.

### Cloning our Git repository
1. Make a directory on the linux server you are using from the command line(terminal): `mkdir application`
2. Get into that directory: `cd application`
3. Clone our repository: `git clone https://charon.cs.uni.edu/birkitaa/4620-predicting-stock-prices.git`
4. Get into that repository: `cd 4620-predicting-stock-prices`
You now have our project repository on your linux box.

### Setup our project(Easy Way)
For our project we made a simple bash script that can setup everything for you to run our code. Or if you want to do it manually you can check our "Run our project(Hard way)". If you already have python3.6 or python3-venv on your server you may want to do the hard way and skip those steps. If unsure still, run this.
1. Access our automation folder `cd automation`
2. Run our make file in that sets up all the "Hard Way" steps `make firstTime`

### Setup our project(Hard way)
Assuming you are in the `/application/4620-predicting-stock-prices` directory:
1. Install python3.6 `sudo apt-get install python3.6`
2. Get python3 venv `sudo apt-get install python3-venv` 
3. Create a virtual environment `python3 -m venv env`
4. Activate your environment `source env/bin/activate`
5. Make sure pip is up to date `pip install --upgrade pip`
6. Get your env up to date with our packages `python3 -m pip install -r requirements.txt`
