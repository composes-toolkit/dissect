# set pythonpath
export PYTHONPATH=/home/thenghia.pham/git/toolkit/src:$PYTHONPATH
export TOOLKIT_DIR=/home/thenghia.pham/git/toolkit
export OUT_DIR=/mnt/cimec-storage-sata/users/thenghia.pham/data/tutorial
export DATA_DIR=/mnt/cimec-storage-sata/users/thenghia.pham/shared/tutorial
export LOG_FILE=$OUT_DIR/log/exercises.log

#**************************************************************************************
echo Step1
echo STARTING BUILDING CORE
export CORE_IN_FILE_PREFIX=CORE_SS.verbnoun.core
export CORE_OUT_DIR=$OUT_DIR/core

# run build core space pipeline
/opt/python/bin/python2.7 $TOOLKIT_DIR/src/pipelines/build_core_space.py -i $DATA_DIR/$CORE_IN_FILE_PREFIX --input_format=pkl -o $CORE_OUT_DIR -w ppmi -s top_sum_2000 -r svd_100 --output_format=dm -l $LOG_FILE

echo FINISHED BUILDING CORE

#**************************************************************************************
echo Step2
echo STARTING PERIPHERAL PIPELINE
export CORE_SPC=CORE_SS.CORE_SS.verbnoun.core.ppmi.top_sum_2000.svd_100.pkl

export PER_RAW_FILE=$DATA_DIR/per.raw.SV
export PER_OUT_DIR=$OUT_DIR/per

# run build peripheral space pipeline
/opt/python/bin/python2.7 $TOOLKIT_DIR/src/pipelines/build_peripheral_space.py -i $PER_RAW_FILE --input_format sm -c $CORE_OUT_DIR/$CORE_SPC -o $PER_OUT_DIR dm -l $LOG_FILE

echo FINISHED PERIPHERAL PIPELINE

#**************************************************************************************
echo step3
echo STARTING TRAINING

export MODEL_DIR=$OUT_DIR/trained
export TRAIN_FILE=$DATA_DIR/ML08_SV_train.txt
export PER_SPC=PER_SS.per.raw.SV.CORE_SS.CORE_SS.verbnoun.core.ppmi.top_sum_2000.svd_100.pkl
export MODEL=lexical_func

# run training pipeline
/opt/python/bin/python2.7 $TOOLKIT_DIR/src/pipelines/train_composition.py -i $TRAIN_FILE -m $MODEL -o $MODEL_DIR -a $CORE_OUT_DIR/$CORE_SPC -p $PER_OUT_DIR/$PER_SPC --regression ridge --intercept True --crossvalidation False --lambda 2.0 -l $LOG_FILE

echo FINISHED TRAINING
#**************************************************************************************
echo step 4
echo STARTING COMPOSING SPACE

export TRNED_MODEL=TRAINED_COMP_MODEL.lexical_func.ML08_SV_train.txt.pkl
export COMP_DIR=$OUT_DIR/composed
export COMP_FILE=$DATA_DIR/ML08nvs_test.txt

# run apply composition pipeline
/opt/python/bin/python2.7 $TOOLKIT_DIR/src/pipelines/apply_composition.py -i $COMP_FILE --load_model $MODEL_DIR/$TRNED_MODEL -o $COMP_DIR -a $CORE_OUT_DIR/$CORE_SPC -l $LOG_FILE

echo FINISHED COMPOSING SPACE
#**************************************************************************************
echo step 5
echo STARTING COMPUTING SIMS

export COMP_SPC=COMPOSED_SS.LexicalFunction.ML08nvs_test.txt.pkl
export SIM_DIR=$OUT_DIR/similarity
export TEST_FILE=$DATA_DIR/ML08data_new.txt

# create output directory for similarity if the directory doesn't exist
if [ ! -d "$SIM_DIR" ]; then
    mkdir $SIM_DIR
fi

# run sim pipeline
 /opt/python/bin/python2.7 $TOOLKIT_DIR/src/pipelines/compute_similarities.py -i $TEST_FILE -s $COMP_DIR/$COMP_SPC -o $SIM_DIR -m cos,lin,dot_prod,euclidean -c 1,2 -l $LOG_FILE

echo FINISH COMPUTE SIMS
#**************************************************************************************
echo step 6
echo STARTING EVAL SIMS

# run evaluation pipeline
/opt/python/bin/python2.7 $TOOLKIT_DIR/src/pipelines/evaluate_similarities.py --in_dir $SIM_DIR -m spearman,pearson -c 3,4 -l $LOG_FILE
echo FINISH EVAL SIMS
