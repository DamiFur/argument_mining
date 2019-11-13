# Script to run the training in all configurations. Must be run from the
# argument mining folder

DATE=$(date +%y-%m-%d-%H-%M)
RESULT_DIRECTORY="/home/mteruel/am/results/echr"
EXPERIMENT_DIRECTORY="ongoing-"$DATE
SEPARATION_LEVEL="paragraph"
DATA_DIR="/home/mteruel/am/data/echr/for_exploration"
echo "******** Starting experiment $DATE"
echo "******** Using self attention "

mkdir $RESULT_DIRECTORY/${SEPARATION_LEVEL}
mkdir $RESULT_DIRECTORY/${SEPARATION_LEVEL}/$EXPERIMENT_DIRECTORY

for PARTITION_DIR in $(compgen -f $DATA_DIR/partition)
do
    PARTITION=$(basename $PARTITION_DIR)
    echo "******** Training on $PARTITION"
    DATASET_NAME=$(compgen -f $PARTITION_DIR/${SEPARATION_LEVEL}/echr-component-classification_komninos_e.p)

    # The model resulting from this command will be saved in --output_dirpath + --experiment_name _model.h5
    KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES=0 python -u experiments/train_SelfAtt_BiLSTM_CNN_CRF.py \
        --dataset $DATASET_NAME \
        --output_dirpath $RESULT_DIRECTORY/${SEPARATION_LEVEL}/$EXPERIMENT_DIRECTORY \
        --experiment_name $PARTITION \
        --char_embedding $1 \
        --char_embedding_size $2 \
        --epochs 50 \
        --classifier $3 \
        --patience 10 \
        --dropout $4 $4 \
        --batch_size $5 \
        --num_units $6 $6 \
        --n_heads $7 \
        --attention_size $8

    if [ $? -ne 0 ]; then { echo "******** Training failed" ; exit 1; } fi
done

echo "********** All experiments completed"
mv $RESULT_DIRECTORY/${SEPARATION_LEVEL}/$EXPERIMENT_DIRECTORY $RESULT_DIRECTORY/${SEPARATION_LEVEL}/$DATE

