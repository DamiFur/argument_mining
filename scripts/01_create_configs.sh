DATA_DIR="../data/echr/for_training"

FILES=( $(ls $DATA_DIR | sed 's/.\{4\}$//' | uniq) )
PARTITION_NUMBER=0
echo $FILES
for TEST_FILENAME in "${FILES[@]}"
do
    echo "Partition $PARTITION_NUMBER"
    echo "Test filename $TEST_FILENAME"
    mkdir "$DATA_DIR/partition$PARTITION_NUMBER"
    mkdir "$DATA_DIR/partition$PARTITION_NUMBER/test"
    cp $DATA_DIR/$TEST_FILENAME.* "$DATA_DIR/partition$PARTITION_NUMBER/test"

    mkdir "$DATA_DIR/partition$PARTITION_NUMBER/dev"
    DEV_FILENAME=$TEST_FILENAME
    while [ $DEV_FILENAME == $TEST_FILENAME ]; do
        rand=$[$RANDOM % ${#FILES[@]}]
        DEV_FILENAME=${FILES[$rand]}
    done
    cp $DATA_DIR/$DEV_FILENAME.* "$DATA_DIR/partition$PARTITION_NUMBER/dev"
    echo "Dev filename $DEV_FILENAME"

    mkdir "$DATA_DIR/partition$PARTITION_NUMBER/train"

    for TRAIN_FILENAME in "${FILES[@]}"
    do
        if [ $TEST_FILENAME != $TRAIN_FILENAME ] && [ $DEV_FILENAME != $TRAIN_FILENAME ]; then
            echo "copying $TRAIN_FILENAME for Training"
            cp $DATA_DIR/$TRAIN_FILENAME* "$DATA_DIR/partition$PARTITION_NUMBER/train"
        fi
    done

    PARTITION_NUMBER=`expr $PARTITION_NUMBER + 1`
done

