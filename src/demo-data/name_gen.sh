#!/usr/bin/env bash

INAME=$1
ONAME="names_${INAME#*_}"

rm -f $ONAME
while read p; do
    md5sum ${ONAME} | awk '{print $1}' >> ${ONAME}
done < $INAME
