#!/bin/bash


IFILE=`basename $1`
IDIR=`dirname $1`

echo $1
echo $IFILE
echo $IDIR

sed -e "s#>FNAME<#$1#g" -e "s#>OFNAME<#$IFILE#g" Factors_Analysis_template.bgr > Factors_Analysis.bgr

xmgrace -batch Factors_Analysis.bgr -nosafe -hardcopy

echo "LAST TIME VALS"
tail -n 1 $1

#gsview "$IFILE".eps

