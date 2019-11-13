SEPARATION_LEVEL="paragraph"
DATA_DIR="../../data/echr/for_training"

cd preprocess

echo "Creating intermidiate representation"

for PARTITION_DIR in $DATA_DIR/partition*
do
    echo "Processing $PARTITION_DIR"
    # These commands create a single list of AnnotatedDocuments from all the annotations
    # in the input_dirpath
    python arg_docs2conll.py --input_dirpath $PARTITION_DIR/train/ --output_file $PARTITION_DIR/train_docs.p
    python arg_docs2conll.py --input_dirpath $PARTITION_DIR/dev/ --output_file $PARTITION_DIR/dev_docs.p
    python arg_docs2conll.py --input_dirpath $PARTITION_DIR/test/ --output_file $PARTITION_DIR/test_docs.p
    # These commands write the AnnotatedDocuments as a single conll file
    echo "Converting partition $PARTITION_DIR to conll format"
    mkdir $PARTITION_DIR/$SEPARATION_LEVEL
    python build_conll.py --input_filename $PARTITION_DIR/train_docs.p --output_filename $PARTITION_DIR/$SEPARATION_LEVEL/train.txt --separation $SEPARATION_LEVEL
    python build_conll.py --input_filename $PARTITION_DIR/dev_docs.p --output_filename $PARTITION_DIR/$SEPARATION_LEVEL/dev.txt --separation $SEPARATION_LEVEL
    python build_conll.py --input_filename $PARTITION_DIR/test_docs.p --output_filename $PARTITION_DIR/$SEPARATION_LEVEL/test.txt --separation $SEPARATION_LEVEL
done
cd ..
