#!/bin/bash
#Author: Jacob Mask
#Simple bash script to run our Neural Network

#Change this to your directory your Makefile is in for us with crontabs and uncomment it
#Example: /home/user/4620-Predicting-Stock-Prices/automation
#cd /home/pnap32032/automation 

#Change this output directory to your logs folder for neural-network outputs
#Example: /home/user/4620-Predicting-Stock-Prices/automation/logs/neural-network-logs/
make runNN > logs/neural-network-logs/"$(date +"%Y_%m_%d_%I_%M_%p").txt" 

#Similar to the example before, please change the email to something else, or add your own email to get updates on ticker predictions
mailx -s "NN Ran" jacobmask@yahoo.com < logs/neural-network-logs/"$(date +"%Y_%m_%d_%I_%M_%p").txt"
