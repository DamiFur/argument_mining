ATTENTION_MODEL=$1
ATTENTION_ACTIVATION=$2

for i in {0..26}; do
    echo "******************* EXPLORING SETTING $i ***************************"

    LSTM_UNITS=(30 50 100)
    rand_lstm_units=${LSTM_UNITS[$((i%3))]}
    echo "LSTM units" $rand_lstm_units

    DROPOUT=(0.1 0.3 0.5)
    rand_dropout=${DROPOUT[$(((i/3)%3))]}
    echo "Dropout" $rand_dropout

    BATCH_SIZE=(30 50 100)
    rand_batch_size=${BATCH_SIZE[$((i/9%3))]}
    echo "Batch size" $rand_batch_size

    bash scripts/08_explore_ukp_attention.sh None 0 \
        Softmax $rand_dropout $rand_batch_size $rand_lstm_units \
        feature_pre sigmoid
done
