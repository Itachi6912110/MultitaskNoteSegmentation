#!/bin/bash

DHEAD="data/TONAS_note/"
AHEAD1="ans/TONAS_onset2/onset_onoffset_"
AHEAD2="ans/TONAS_offset2/onset_offset_"
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

MDIR="model/${NORM}${WS}_l${HL1}h${HS1}b${BIDIR1}_e${EPOCHS}b${BATCH}_${FEAT1}"
EMFILE1="${MDIR}/onoffset_attn_${NORM}_onenc_k${WS}l${HL1}h${HS1}b${BIDIR1}e${EPOCHS}b${BATCH}_${FEAT1}"
DMFILE1="${MDIR}/onoffset_attn_${NORM}_ondec_k${WS}l${HL1}h${HS1}b${BIDIR1}e${EPOCHS}b${BATCH}_${FEAT1}"
LFILE="loss/onoffset_attn_${NORM}${WS}_l${HL1}h${HS1}b${BIDIR1}_e${EPOCHS}b${BATCH}_${FEAT1}.csv"

echo -e "Training OnOffset Model Exp2 Info:"
echo -e "HS1=${HS1} HL1=${HL1} BIDIR1=${BIDIR1} NORM=${NORM}"
echo -e "WS=${WS} SE=${SE}"
echo -e "EPOCHS=${EPOCHS} BATCH=${BATCH} FEAT1=${FEAT1}"
echo -e "Onset Encoder Model: ${EMFILE1}"
echo -e "Onset Decoder Model: ${DMFILE1}"
echo -e "Loss: ${LFILE}"

mkdir -p ${MDIR}

for e in $(seq 0 $((${EPOCHS}-1)))
do
    for num in $(seq 1 82)
    do
        python3 onoffset_exp2.py -d1 ${DHEAD}${num}_${FEAT1} -a1 ${AHEAD1}${num} -a2 ${AHEAD2}${num} -em1 ${EMFILE1} \
        -dm1 ${DMFILE1} -p ${num} -e ${e} -l ${LR} \
        --hs1 ${HS1} --hl1 ${HL1} --window-size ${WS} --single-epoch ${SE} --bi1 ${BIDIR1} \
        --loss-record ${LFILE} --batch-size ${BATCH} --norm ${NORM} --feat1 ${FEAT_NUM1}
    done
    
done