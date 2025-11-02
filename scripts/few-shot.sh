#!/bin/bash

cd ..

# Custom config
DATA=/path/to/your/dataset

TRAINER=CAP


CFG=vit_b32_ep50_ctxv1_few_shot
CTP=end 
SHOTS=${2:-16}
CSC=False 
FOLDER=output
IND=0  
for NCTX in 6; do
    for NCTX_V in 4; do
        for DATASET in caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft dtd eurosat ucf101; do
            for SEED in 1 2 3; do
                DIR=${FOLDER}/few-shot/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
                
                if [ -d "$DIR" ]; then
                    echo "Results are available in ${DIR}. Skip this job"
                else
                    echo "Run this job and save the output to ${DIR}"
                    CUDA_VISIBLE_DEVICES=$1 python train.py \
                        --root ${DATA} \
                        --seed ${SEED} \
                        --trainer ${TRAINER} \
                        --dataset-config-file configs/datasets/${DATASET}.yaml \
                        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                        --output-dir ${DIR} \
                        TRAINER.COOP.N_CTX ${NCTX} \
                        TRAINER.COOP.N_CTX_V ${NCTX_V} \
                        TRAINER.COOP.CSC ${CSC} \
                        TRAINER.COOP.L_IND ${IND} \
                        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
                        DATASET.NUM_SHOTS ${SHOTS}
                fi
            done
        done
    done
done

CFG=vit_b32_ep10_ctxv1_few_shot
for NCTX in 6; do
    for NCTX_V in 4; do
        for DATASET in imagenet sun397; do
            for SEED in 1 2 3; do
                DIR=${FOLDER}/few-shot/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
                
                if [ -d "$DIR" ]; then
                    echo "Results are available in ${DIR}. Skip this job"
                else
                    echo "Run this job and save the output to ${DIR}"
                    CUDA_VISIBLE_DEVICES=$1 python train.py \
                        --root ${DATA} \
                        --seed ${SEED} \
                        --trainer ${TRAINER} \
                        --dataset-config-file configs/datasets/${DATASET}.yaml \
                        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                        --output-dir ${DIR} \
                        TRAINER.COOP.N_CTX ${NCTX} \
                        TRAINER.COOP.N_CTX_V ${NCTX_V} \
                        TRAINER.COOP.CSC ${CSC} \
                        TRAINER.COOP.L_IND ${IND} \
                        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
                        DATASET.NUM_SHOTS ${SHOTS}
                fi
            done
        done
    done
done

