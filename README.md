# Stock Prediction Application

## Updates
Currently turning this into a LAMP stack website for my Master's Program

## First time Installation and setup guide
This can be done having a virtualbox running ubuntu 18.04(preferred), a guide can be shown here:
https://brb.nci.nih.gov/seqtools/installUbuntu.html
Install the Desktop image of ubuntu here: https://releases.ubuntu.com/18.04/ and put that into the virtual box if that is your plan to use.

This can still be done other ways, for our project we had a private virtual linux server. The student server cannot work our project without changes to some files, since the student server is on python3.5 and we have it on 3.6.

### Cloning our Git repository
1. Make a directory on the linux server you are using from the command line(terminal): `mkdir application`
2. Get into that directory: `cd application`
3. Clone our repository: `git clone https://github.com/jacobmask/Stock-Predictions.git`
4. Get into that repository: `cd Stock-Predictions`
You now have our project repository on your linux box.

### Setup our project(Easy Way)
For our project we made a simple bash script that can setup everything for you to run our code. Or if you want to do it manually you can check our "Run our project(Hard way)". If you already have python3.6 or python3-venv on your server you may want to do the hard way and skip those steps. If unsure still, run this.
Assuming you are in the `/application/4620-predicting-stock-prices` directory:
1. Access our automation folder `cd automation`
2. Run our make file in that sets up all the "Hard Way" steps automatically `make firstTime`

### Setup our project(Hard Way)
Assuming you are in the `/application/4620-predicting-stock-prices` directory:
1. Install python3.6 `sudo apt-get install python3.6`
2. Get python3 venv `sudo apt-get install python3-venv` 
3. Create a virtual environment `python3 -m venv env`
4. Activate your environment `source env/bin/activate`
5. Make sure pip is up to date `pip install --upgrade pip`
6. Get your env up to date with our packages `python3 -m pip install -r requirements.txt`
NOTE: To close out of an environment, run `deactivate`


## How to run our code properly
Our code is all in python files, with some automation scripts in bash and makefiles. You can run our code yourself with our "Run our code yourself" guide, or have it automated with our "Run our code automatically" guide. The difference between the two is having bash scripts for the automatic guide, since we wanted our code to run automously. This can also be much simpler to run for beginners to linux.

### Run Our Code yourself
Our code has GPU and non GPU versions to run. If you want a graph(GPU version) that is our `./` directory. If you don't have a gpu use our `./automation/` directory.
1. Enable your environment `source env/bin/activate`
2. (OPTIONAL) Update our config.py with stock tickers you wish to use. Edit it with an editor like nano 
   `nano config.py` -- follow similar formatting used in that file. Our stock ticker list currently contains stocks
   that have analyst recommendations for better predictions. CTRL+X to exit, type Y and enter to save.
(OPTIONAL) - You can run our `run-app.py` file and skip steps 3 and 4 if you wish. You can run it with `python3 run-app.py`
3. Update the active stocks `python3 data-clean.py` this generates CSV file outputs of each stock in config.py to 
   the StockCSVRecs folder with current data.
4. Run the training model `python3 neural-network.py` this will print some warnings(ignore those) as well as
   predicted prices for end of day versus current prices.

### Run Our Code automatically
Our automatically ran code is accessed in our `./automation` directory.
1. `cd automation`
2. (OPTIONAL) Update our config.py with stock tickers you wish to use. Edit it with an editor like nano 
   `nano config.py` -- follow similar formatting used in that file. Our stock ticker list currently contains stocks
   that have analyst recommendations for better predictions. CTRL+X to exit, type Y and enter to save.
3. `./newDayScript.sh` this takes awhile to run, give it at least a minute. This runs our stock pulling for current
   data.
4. `./runNNScript.sh` this creates end of day predictions and compares them to actual prices. The output is created in the "/automation/logs/neural-network-logs/" directory. Labeled as "Year_month_day_hour_minute.txt". This also prints errors when running in terminal if you don't have a GPU. That can be ignored, the output is still the same in the txt file.

#### Have some fun with crontab
We also had crontab running on our linux server, it can be viewed here: "./automation/crontabexample.txt". Crontab editting is in it's own file and needs to be created per linux server, so we have it in txt to show as an example. We had this run our stock puller and neural network 3 times a day to check for validation. We had our `./runNNScript.sh` email us everytime it was executed with the output. Making it easier to view from mobile, rather than having to login to the linux server everyday.

NOTE: Our bash scripts work when you run them directly, but if you want crontabs to work running them you will need to edit some code. In the `newDayScript.sh` file you need the full path on line 13 for the output `/home/pnap32032/automation/logs/newDayLogs.txt`to `directorysbefore/automation/logs/newDayLogs.txt`. `runNNScript.sh` also needs full path directories for their last parts of code. This is an issue with crontab running from root I believe and has a root directory while running these files. There are comments on both files on what code areas need changed. This is not needed, but recommended to use with crontabs.


# Licensing 
We follow the [MIT](https://choosealicense.com/licenses/mit/) license formatting.
