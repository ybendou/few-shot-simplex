# banana-penguin
Repo for few-shot learning evaluation. 

Run the command to add to path :
```
 export PYTHONPATH=<path>/banana-penguin:$PYTHONPATH
```

- Run our method:
```
    python run_closest_summet.py --features-path \
        "['<path>/miniAS50backbone11.pt', '<path>/miniAS100backbone11.pt', '<path>/miniAS150backbone11.pt', '<path>/miniAS200backbone11.pt']" \
        --features-base-path '<path>/minifeaturesAS1.pt11' \
        --centroids-file '<path>/miniImagenetAS200noPrepLamda05.pickle' --lamda-mix 0.25 --n-runs 100000 --preprocessing 'ME';
```


- Run the baseline (AS):
```
    python run_closest_summet.py --features-path \
        "['<path>/miniAS50backbone11.pt', '<path>/miniAS100backbone11.pt', '<path>/miniAS150backbone11.pt', '<path>/miniAS200backbone11.pt']" \
        --features-base-path '<path>/minifeaturesAS1.pt11' \
        --centroids-file '<path>/miniImagenetAS200noPrepLamda05.pickle' --lamda-mix 0 --n-runs 100000 --preprocessing 'ME';
```

- To extract simplex summits: 
```
python run_centroid_extraction.py --features-path \
    "['<path>/miniAS0backbone11.pt', '<path>/miniAS1backbone11.pt', '<path>/miniAS2backbone11.pt', '<path>/miniAS3backbone11.pt']" \
    --centroids-file '<path>/miniImagenetAS1000_0123_noPrep_Simplex0.05.pickle' --extraction 'simplex' --lamda-reg 0.05 --thresh-elbow 1.5;
```