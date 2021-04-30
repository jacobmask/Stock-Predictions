#!/bin/bash
#Author: Jacob Mask
#Simple bash script to run our new Day stock puller


#Change this to your directory your Makefile is in for use with crontabs and uncomment it
#Example: /home/user/4620-Predicting-Stock-Prices/automation 
#cd /home/pnap32032/automation
make newDay

#Change this output directory to your logs
#Example: /home/user/4620-Predicting-Stock-Prices/automation/logs/newDayLogs.txt
echo "$(date +"%Y_%m_%d_%I_%M_%p")" >> logs/newDayLogs.txt
