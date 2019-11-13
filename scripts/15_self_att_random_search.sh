HEADS=$1

echo "Attention heads " $HEADS

for i in 1 2 3 4 5; do
    echo "******************* EXPLORING SETTING $i ***************************"
    CHAR_EMB=(lstm None cnn)
    rand_char_emb=${CHAR_EMB[$[$RANDOM % ${#CHAR_EMB[@]}]]}
    echo "Char embedding type" $rand_char_emb

    CHAR_EMB_SIZE=(16 32 64)
    rand_char_emb_size=${CHAR_EMB_SIZE[$[$RANDOM % ${#CHAR_EMB_SIZE[@]}]]}
    echo "Char embedding size" $rand_char_emb_size

    CLASSIFIER=(CRF CRF Softmax)
    rand_classifier=${CLASSIFIER[$[$RANDOM % ${#CLASSIFIER[@]}]]}
    echo "Classifier" $rand_classifier

    LSTM_UNITS=(30 50 100 200)
    rand_lstm_units=${LSTM_UNITS[$[$RANDOM % ${#LSTM_UNITS[@]}]]}
    echo "LSTM units" $rand_lstm_units

    DROPOUT=(0.1 0.2 0.3 0.4 0.5)
    rand_dropout=${DROPOUT[$[$RANDOM % ${#DROPOUT[@]}]]}
    echo "Dropout" $rand_dropout

    BATCH_SIZE=(30 50 50 100 200)
    rand_batch_size=${BATCH_SIZE[$[$RANDOM % ${#BATCH_SIZE[@]}]]}
    echo "Batch size" $rand_batch_size

    ATTENTION_SIZE=(32 64 128)
    rand_attention_size=${ATTENTION_SIZE[$[$RANDOM % ${#ATTENTION_SIZE[@]}]]}
    echo "Attention size" $rand_attention_size

    bash scripts/14_train_echr_self.sh $rand_char_emb $rand_char_emb_size \
        $rand_classifier $rand_dropout $rand_batch_size $rand_lstm_units \
        $HEADS $rand_attention_size
done
