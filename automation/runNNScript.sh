#!/bin/bash
make runNN > logs/neural-network-logs/"$(date +"%Y_%m_%d_%I_%M_%p").txt"
mailx -s "NN Ran" jacobmask@yahoo.com < logs/neural-network-logs/"$(date +"%Y_%m_%d_%I_%M_%p").txt"
