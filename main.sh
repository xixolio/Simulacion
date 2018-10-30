#!/bin/bash
#PBS -l cput=1000:00:00
#PBS -l walltime=1000:00:00
#PBS -l mem=4gb

use anaconda3
use gcc48
KERAS_BACKEND=tensorflow python /user/i/iaraya/Simulacion/periodic_mu_sigma_model.py $1 $2 $3 $4 $5 $6
