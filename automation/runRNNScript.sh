#!/bin/bash
make runRNN > rnnLogs/"$(date +"%Y_%m_%d_%I_%M_%p").txt"
mailx -s "RNN Ran" jacobmask@yahoo.com < rnnLogs/"$(date +"%Y_%m_%d_%I_%M_%p").txt"
