

I will make this pretty later. 

You can still run any of the models originally included in the library, as well as any of the HMAX models in timm/models/HMAX.py

To run with contrastive loss: use CHMAX as the model. Pass the following model kwargs: 
- ip_scale_bands
    This is the number of scale bands (one more than the number of images in the pyramid)
- classifier_input_size
    Input size to the classifier (explicit parameter because it is not currently being calculated automatically)
- hmax_type
    This can only be "full" or "bypass" at the moment. 

Additionally, you MUST specify the contrastive loss lambda as an argument in the run script with --cl-lambda [value]. The default is 0 (which means the cl term will be discarded).


No other models will currently run with contrastive loss. 

See run_hmax.sh for an example of a launch script.
