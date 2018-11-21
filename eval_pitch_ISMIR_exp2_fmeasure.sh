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
TESTGROUP=G1

THRESHOLD=0.5

MDIR="model/${NORM}${WS}_l${HL1}h${HS1}b${BIDIR1}_e${EPOCHS}b${BATCH}_${FEAT}_${TESTGROUP}"
EMFILE1="${MDIR}/onoffset_attn_${NORM}_onenc_k${WS}l${HL1}h${HS1}b${BIDIR1}e${EPOCHS}b${BATCH}_${FEAT}"
DMFILE1="${MDIR}/onoffset_attn_${NORM}_ondec_k${WS}l${HL1}h${HS1}b${BIDIR1}e${EPOCHS}b${BATCH}_${FEAT}"
EFILE="output/single/onoffset_ISMIR2014_${NORM}k${WS}_l${HL1}h${HS1}b{BIDIR1}_e${EPOCHS}b${BATCH}_${FEAT}_${TESTGROUP}.csv"
VFILE="output/total/onoffset_ISMIR2014_${NORM}k${WS}_l${HL1}h${HS1}b{BIDIR1}_e${EPOCHS}b${BATCH}_${FEAT}_${TESTGROUP}.csv"
TROUTDIR="output/est"

echo -e "Evaluating OnOffset Model ISMIR2014 Info:"
echo -e "Group: ${TESTGROUP}"
echo -e "HS1=${HS1} HL1=${HL1} BIDIR1=${BIDIR1} NORM=${NORM}"
echo -e "HS2=${HS2} HL2=${HL2}"
echo -e "WS=${WS} SE=${SE}"
echo -e "EPOCHS=${EPOCHS} BATCH=${BATCH} FEAT=${FEAT}"
echo -e "Onset Encoder Model: ${EMFILE1}"
echo -e "Onset Decoder Model: ${DMFILE1}"
echo -e "Offset Decoder Model: ${DMFILE2}"

echo -e "Start Evaluation on ISMIR2014 Validation Set"

mkdir -p ${TROUTDIR}

if [ "${TESTGROUP}" == "G1" ]; then
    for num in 1 2 12 13 25 26 27
    do
        python3 eval_pitch_exp2_fmeasure.py -d ${DHEAD}${num}_${FEAT} -a ${AHEAD}${num}.GroundTruth -of ${TROUTDIR}/${num}_CV_test -pf ${PHEAD}${num}_P -em1 ${EMFILE1} -dm1 ${DMFILE1} -p ${num} -ef ${EFILE} -tf ${VFILE} -l ${LR} \
        --hs1 ${HS1} --hl1 ${HL1} --ws ${WS} --single-epoch ${SE} --bidir1 ${BIDIR1} --norm ${NORM} --feat ${FEAT_NUM} --threshold ${THRESHOLD}
    done
fi
if [ "${TESTGROUP}" == "G2" ]; then
    for num in 3 4 14 15 28 29 30
    do
        python3 eval_pitch_exp2_fmeasure.py -d ${DHEAD}${num}_${FEAT} -a ${AHEAD}${num}.GroundTruth -of ${TROUTDIR}/${num}_CV_test -pf ${PHEAD}${num}_P -em1 ${EMFILE1} -dm1 ${DMFILE1} -p ${num} -ef ${EFILE} -tf ${VFILE} -l ${LR} \
        --hs1 ${HS1} --hl1 ${HL1} --ws ${WS} --single-epoch ${SE} --bidir1 ${BIDIR1} --norm ${NORM} --feat ${FEAT_NUM} --threshold ${THRESHOLD}
    done
fi
if [ "${TESTGROUP}" == "G3" ]; then
    for num in 5 6 16 17 18 31 32 33
    do
        python3 eval_pitch_exp2_fmeasure.py -d ${DHEAD}${num}_${FEAT} -a ${AHEAD}${num}.GroundTruth -of ${TROUTDIR}/${num}_CV_test -pf ${PHEAD}${num}_P -em1 ${EMFILE1} -dm1 ${DMFILE1} -p ${num} -ef ${EFILE} -tf ${VFILE} -l ${LR} \
        --hs1 ${HS1} --hl1 ${HL1} --ws ${WS} --single-epoch ${SE} --bidir1 ${BIDIR1} --norm ${NORM} --feat ${FEAT_NUM} --threshold ${THRESHOLD}
    done
fi
if [ "${TESTGROUP}" == "G4" ]; then
    for num in 7 8 19 20 21 34 35 36
    do
        python3 eval_pitch_exp2_fmeasure.py -d ${DHEAD}${num}_${FEAT} -a ${AHEAD}${num}.GroundTruth -of ${TROUTDIR}/${num}_CV_test -pf ${PHEAD}${num}_P -em1 ${EMFILE1} -dm1 ${DMFILE1} -p ${num} -ef ${EFILE} -tf ${VFILE} -l ${LR} \
        --hs1 ${HS1} --hl1 ${HL1} --ws ${WS} --single-epoch ${SE} --bidir1 ${BIDIR1} --norm ${NORM} --feat ${FEAT_NUM} --threshold ${THRESHOLD}
    done
fi
if [ "${TESTGROUP}" == "G5" ]; then
    for num in 9 10 11 22 23 24 37 38
    do
        python3 eval_pitch_exp2_fmeasure.py -d ${DHEAD}${num}_${FEAT} -a ${AHEAD}${num}.GroundTruth -of ${TROUTDIR}/${num}_CV_test -pf ${PHEAD}${num}_P -em1 ${EMFILE1} -dm1 ${DMFILE1} -p ${num} -ef ${EFILE} -tf ${VFILE} -l ${LR} \
        --hs1 ${HS1} --hl1 ${HL1} --ws ${WS} --single-epoch ${SE} --bidir1 ${BIDIR1} --norm ${NORM} --feat ${FEAT_NUM} --threshold ${THRESHOLD}
    done
fi