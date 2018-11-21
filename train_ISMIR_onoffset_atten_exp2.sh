#!/bin/bash

DHEAD="data/ISMIR2014_note/"
AHEAD1="ans/ISMIR2014_onset2/onset_"
AHEAD2="ans/ISMIR2014_offset2/offset_"
BASE_LR=0.001
SIZE_LR=$1
LR=$(echo "${BASE_LR} * ${SIZE_LR}" | bc)
HS1=$2
HL1=$3
WS=$4
SE=$5
BIDIR1=$6
NORM=$7

EPOCHS=$8
BATCH=$9
FEAT1=${10}
FEAT_NUM1=${11}
TRAINCOUNT=71

TESTGROUP=${12}

MDIR="model/${NORM}${WS}_l${HL1}h${HS1}b${BIDIR1}_e${EPOCHS}b${BATCH}_${FEAT1}_${TESTGROUP}"
EMFILE1="${MDIR}/onoffset_attn_${NORM}_onenc_k${WS}l${HL1}h${HS1}b${BIDIR1}e${EPOCHS}b${BATCH}_${FEAT1}"
DMFILE1="${MDIR}/onoffset_attn_${NORM}_ondec_k${WS}l${HL1}h${HS1}b${BIDIR1}e${EPOCHS}b${BATCH}_${FEAT1}"
LFILE="loss/onoffset_attn_${NORM}${WS}_l${HL1}h${HS1}b${BIDIR1}_e${EPOCHS}b${BATCH}_${FEAT1}_${TESTGROUP}.csv"

echo -e "Training OnOffset Model V3 on ISMIR2014 Info:"
echo -e "Test Group=${TESTGROUP}"
echo -e "HS1=${HS1} HL1=${HL1} BIDIR1=${BIDIR1} NORM=${NORM}"
echo -e "WS=${WS} SE=${SE}"
echo -e "EPOCHS=${EPOCHS} BATCH=${BATCH} FEAT1=${FEAT1}"
echo -e "Onset Encoder Model: ${EMFILE1}"
echo -e "Onset Decoder Model: ${DMFILE1}"
echo -e "Loss: ${LFILE}"

mkdir -p ${MDIR}

for e in $(seq 0 $((${EPOCHS}-1)))
do
    if [ "${TESTGROUP}" != "G1" ]; then
        for num in 1 2 12 13 25 26 27
        do
            python3 onoffset_exp2.py -d1 ${DHEAD}${num}_${FEAT1} -a1 ${AHEAD1}${num} -a2 ${AHEAD2}${num} -em1 ${EMFILE1} \
            -dm1 ${DMFILE1} -p ${num} -e ${e} -l ${LR} \
            --hs1 ${HS1} --hl1 ${HL1} --window-size ${WS} --single-epoch ${SE} --bi1 ${BIDIR1} \
            --loss-record ${LFILE} --batch-size ${BATCH} --norm ${NORM} --feat1 ${FEAT_NUM1}
        done
    fi

    if [ "${TESTGROUP}" != "G2" ]; then
        for num in 3 4 14 15 28 29 30
        do
        if [ -f ${EMFILE1} ]; then

            python3 onoffset_exp2.py -d1 ${DHEAD}${num}_${FEAT1} -a1 ${AHEAD1}${num} -a2 ${AHEAD2}${num} -em1 ${EMFILE1} \
            -dm1 ${DMFILE1} -p ${num} -e ${e} -l ${LR} \
            --hs1 ${HS1} --hl1 ${HL1} --window-size ${WS} --single-epoch ${SE} --bi1 ${BIDIR1} \
            --loss-record ${LFILE} --batch-size ${BATCH} --norm ${NORM} --feat1 ${FEAT_NUM1}
        else
            python3 onoffset_exp2.py -d1 ${DHEAD}${num}_${FEAT1} -a1 ${AHEAD1}${num} -a2 ${AHEAD2}${num} -em1 ${EMFILE1} \
            -dm1 ${DMFILE1} -p 1 -e ${e} -l ${LR} \
            --hs1 ${HS1} --hl1 ${HL1} --window-size ${WS} --single-epoch ${SE} --bi1 ${BIDIR1} \
            --loss-record ${LFILE} --batch-size ${BATCH} --norm ${NORM} --feat1 ${FEAT_NUM1}
        fi

        done
    fi

    if [ "${TESTGROUP}" != "G3" ]; then
        for num in 5 6 16 17 18 31 32 33
        do
            python3 onoffset_exp2.py -d1 ${DHEAD}${num}_${FEAT1} -a1 ${AHEAD1}${num} -a2 ${AHEAD2}${num} -em1 ${EMFILE1} \
            -dm1 ${DMFILE1} -p ${num} -e ${e} -l ${LR} \
            --hs1 ${HS1} --hl1 ${HL1} --window-size ${WS} --single-epoch ${SE} --bi1 ${BIDIR1} \
            --loss-record ${LFILE} --batch-size ${BATCH} --norm ${NORM} --feat1 ${FEAT_NUM1}
        done
    fi

    if [ "${TESTGROUP}" != "G4" ]; then
        for num in 7 8 19 20 21 34 35 36
        do
            python3 onoffset_exp2.py -d1 ${DHEAD}${num}_${FEAT1} -a1 ${AHEAD1}${num} -a2 ${AHEAD2}${num} -em1 ${EMFILE1} \
            -dm1 ${DMFILE1} -p ${num} -e ${e} -l ${LR} \
            --hs1 ${HS1} --hl1 ${HL1} --window-size ${WS} --single-epoch ${SE} --bi1 ${BIDIR1} \
            --loss-record ${LFILE} --batch-size ${BATCH} --norm ${NORM} --feat1 ${FEAT_NUM1}
        done
    fi

    if [ "${TESTGROUP}" != "G5" ]; then
        for num in 9 10 11 22 23 24 37 38
        do
            python3 onoffset_exp2.py -d1 ${DHEAD}${num}_${FEAT1} -a1 ${AHEAD1}${num} -a2 ${AHEAD2}${num} -em1 ${EMFILE1} \
            -dm1 ${DMFILE1} -p ${num} -e ${e} -l ${LR} \
            --hs1 ${HS1} --hl1 ${HL1} --window-size ${WS} --single-epoch ${SE} --bi1 ${BIDIR1} \
            --loss-record ${LFILE} --batch-size ${BATCH} --norm ${NORM} --feat1 ${FEAT_NUM1}
        done
    fi

done