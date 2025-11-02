#!/bin/bash


cd ..


DATA=/path/to/your/dataset

TRAINER=CAP

CFG=vit_b32_ep3_ctxv1_cross_dataset
CTP=end        
NCTX=6        
NCTX_V=4        
SHOTS=16        
CSC=False    
FOLDER=output
IND=0        

# Loop through configurations
for NCTX in 6; do
    for NCTX_V in 4; do
        for DATASET in imagenet_a imagenet_r imagenet_sketch imagenetv2; do
            for SEED in 1 2 3; do
                LOADEP=3
                MODEL_DIR=${FOLDER}/train_all/imagenet/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
                DIR=${FOLDER}/dg/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
                if [ -d "$DIR" ]; then
                    echo "Results are available in ${DIR}. Skip this job"
                else
                    echo "Run this job and save the output to ${DIR}"
                    CUDA_VISIBLE_DEVICES=$1 python train.py \
                        --root ${DATA} \
                        --seed ${SEED} \
                        --trainer ${TRAINER} \
                        --dataset-config-file configs/datasets/${DATASET}.yaml \
                        --config-file configs/trainers/${TRAINER}/xd.yaml \
                        --output-dir ${DIR} \
                        --model-dir ${MODEL_DIR} \
                        --load-epoch ${LOADEP} \
                        --eval-only \
                        TRAINER.COOP.N_CTX ${NCTX} \
                        TRAINER.COOP.N_CTX_V ${NCTX_V} \
                        TRAINER.COOP.L_IND ${IND} \
                        TRAINER.COOP.CSC ${CSC} \
                        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
                        DATASET.NUM_SHOTS ${SHOTS}
                fi
            done
        done
    done
done
