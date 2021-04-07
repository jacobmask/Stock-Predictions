#!/bin/bash
cd /home/pnap32032/automation/
make newDay
echo "$(date +"%Y_%m_%d_%I_%M_%p")" >> /home/pnap32032/automation/newDayLogs.txt
