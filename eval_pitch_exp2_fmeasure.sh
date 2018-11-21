#!/bin/bash

DHEAD="data/ISMIR2014_note/"
AHEAD="ans/ISMIR2014_ans/"
PHEAD="pitch/ISMIR2014/"
SIZE_LR=1
BASE_LR=0.001
LR=$(echo "${BASE_LR} * ${SIZE_LR}" | bc)
HS1=100
HL1=3
WS=9
SE=5
BIDIR1=1
NORM=ln

EPOCHS=80
BATCH=10
FEAT=SN_SF1_SIN_SF1_ZN_F9
FEAT_NUM=9

THRESHOLD=0.5

MDIR="model/${NORM}${WS}_l${HL1}h${HS1}b${BIDIR1}_e${EPOCHS}b${BATCH}_${FEAT}"
EMFILE1="${MDIR}/onoffset_attn_${NORM}_onenc_k${WS}l${HL1}h${HS1}b${BIDIR1}e${EPOCHS}b${BATCH}_${FEAT}"
DMFILE1="${MDIR}/onoffset_attn_${NORM}_ondec_k${WS}l${HL1}h${HS1}b${BIDIR1}e${EPOCHS}b${BATCH}_${FEAT}"
EFILE="output/single/onoffset_ISMIR2014_${NORM}k${WS}_l${HL1}h${HS1}b{BIDIR1}_e${EPOCHS}b${BATCH}_${FEAT}.csv"
VFILE="output/total/onoffset_ISMIR2014_${NORM}k${WS}_l${HL1}h${HS1}b{BIDIR1}_e${EPOCHS}b${BATCH}_${FEAT}.csv"
TROUTDIR="output/est"

echo -e "Evaluating OnOffset Model Info:"
echo -e "HS1=${HS1} HL1=${HL1} BIDIR1=${BIDIR1} NORM=${NORM}"
echo -e "HS2=${HS2} HL2=${HL2}"
echo -e "WS=${WS} SE=${SE}"
echo -e "EPOCHS=${EPOCHS} BATCH=${BATCH} FEAT=${FEAT}"
echo -e "Onset Encoder Model: ${EMFILE1}"
echo -e "Onset Decoder Model: ${DMFILE1}"

echo -e "Start Evaluation on ISMIR2014 Validation Set"

mkdir -p ${TROUTDIR}

for num in $(seq 1 38)
do
    python3 eval_pitch_exp2_fmeasure.py -d ${DHEAD}${num}_${FEAT} -a ${AHEAD}${num}.GroundTruth -pf ${PHEAD}${num}_P -em1 ${EMFILE1} -dm1 ${DMFILE1} -p ${num} -ef ${EFILE} -tf ${VFILE} -l ${LR} \
    --hs1 ${HS1} --hl1 ${HL1} --ws ${WS} --single-epoch ${SE} --bidir1 ${BIDIR1} --norm ${NORM} --feat ${FEAT_NUM} --threshold ${THRESHOLD} -of ${TROUTDIR}/${num}_CD_test
done