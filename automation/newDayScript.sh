#!/bin/bash
make newDay
echo "$(date +"%Y_%m_%d_%I_%M_%p")" >> logs/newDayLogs.txt
