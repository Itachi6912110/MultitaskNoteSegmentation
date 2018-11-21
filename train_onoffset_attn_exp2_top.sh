#!/bin/bash

SIZE_LR=1
HS1=100
HL1=3
WS=9
SE=5
BIDIR1=1
NORM=ln

EPOCHS=80
BATCH=10
FEAT1=SN_SF1_SIN_SF1_ZN_F9
FEAT_NUM1=9

LOG_FILE="log/${NORM}${WS}_l${HL1}h${HS1}b${BIDIR1}_e${EPOCHS}b${BATCH}_${FEAT1}.log"

echo -e "Saving Record to ${LOG_FILE}"

bash train_TONAS_onoffset_atten_exp2.sh ${SIZE_LR} ${HS1} ${HL1} ${WS} ${SE} ${BIDIR1} ${NORM} ${EPOCHS} ${BATCH} ${FEAT1} ${FEAT_NUM1} | tee ${LOG_FILE}

echo -e "Saving Record to ${LOG_FILE}"