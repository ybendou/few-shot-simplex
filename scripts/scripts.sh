
echo "Running experiments for COCO" >> $MYSPACE/experiments/simplex/coco/results/cocoval2017_dino_base8_224px_AS200coco_test_features_simplex005_dict.txt;
for x in {0..20}; do
    y=`bc <<< "scale=2; $x/20"`
    python run_closest_summet_coco.py --features-path $MYSPACE/experiments/simplex/coco/features/cocoval2017_dino_base8_224px_AS200coco_test_features.pt --centroids-file $MYSPACE/experiments/simplex/coco/centroids/cocoval2017_dino_base8_224px_AS200coco_test_features_simplex005_dict.pickle --n-ways 5 --n-runs 1000 --n-queries 15 --lamda-mix $y --n-shots 1 >> $MYSPACE/experiments/simplex/coco/results/cocoval2017_dino_base8_224px_AS200coco_test_features_simplex005_dict.txt;
done

echo "Running experiments for COCO" >> $MYSPACE/experiments/simplex/coco/results/cocoval2017_dino_base8_224px_AS200coco_test_features_simplex01_dict.txt;
for x in {0..20}; do
    y=`bc <<< "scale=2; $x/20"`
    python run_closest_summet_coco.py --features-path $MYSPACE/experiments/simplex/coco/features/cocoval2017_dino_base8_224px_AS200coco_test_features.pt --centroids-file $MYSPACE/experiments/simplex/coco/centroids/cocoval2017_dino_base8_224px_AS200coco_test_features_simplex01_dict.pickle --n-ways 5 --n-runs 1000 --n-queries 15 --lamda-mix $y --n-shots 1 >> $MYSPACE/experiments/simplex/coco/results/cocoval2017_dino_base8_224px_AS200coco_test_features_simplex01_dict.txt;
done

echo "Running experiments for COCO" >> $MYSPACE/experiments/simplex/coco/results/cocoval2017_dino_base8_224px_AS200coco_test_features_simplex001_dict.txt;
for x in {0..20}; do
    y=`bc <<< "scale=2; $x/20"`
    python run_closest_summet_coco.py --features-path $MYSPACE/experiments/simplex/coco/features/cocoval2017_dino_base8_224px_AS200coco_test_features.pt --centroids-file $MYSPACE/experiments/simplex/coco/centroids/cocoval2017_dino_base8_224px_AS200coco_test_features_simplex001_dict.pickle --n-ways 5 --n-runs 1000 --n-queries 15 --lamda-mix $y --n-shots 1 >> $MYSPACE/experiments/simplex/coco/results/cocoval2017_dino_base8_224px_AS200coco_test_features_simplex001_dict.txt;
done

echo "Running experiments for COCO" >> $MYSPACE/experiments/simplex/coco/results/cocoval2017_dino_base8_224px_AS200coco_test_features_simplex015_dict.txt;
for x in {0..20}; do
    y=`bc <<< "scale=2; $x/20"`
    python run_closest_summet_coco.py --features-path $MYSPACE/experiments/simplex/coco/features/cocoval2017_dino_base8_224px_AS200coco_test_features.pt --centroids-file $MYSPACE/experiments/simplex/coco/centroids/cocoval2017_dino_base8_224px_AS200coco_test_features_simplex015_dict.pickle --n-ways 5 --n-runs 1000 --n-queries 15 --lamda-mix $y --n-shots 1 >> $MYSPACE/experiments/simplex/coco/results/cocoval2017_dino_base8_224px_AS200coco_test_features_simplex015_dict.txt;
done

