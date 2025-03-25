## DL parallel optimization
PredTOP can use the latency prediction based technique to find the optimal parallel execution plan for DL training. In order to run the optimization based on PredTOP's latency prediction use the following command.
```bash
cd benchmark
SAVE_MODELS_DIR='<directory_to_save_the_prediction_models>' python run_benchmark.py --model=<model_name>
```

This takes two options, the first is the environment variable `SAVE_MODELS_DIR`. Set this variable to a path which will be used to store the trained prediction models. If you decide to predict the latency of a specific plan or rerun the optimization process by expediting the training, these trained models can be used. If you have already ran this command, you can use the following command to use the pretrained models.

```bash
cd benchmark
SAVED_MODELS_DIR='<path_to_saved_models>' python run_benchmark.py --model=<model_name>
```

The second option, `--model` specifies which benchmark to run. You can set it to either `moe` or `gpt` based on which benchmark you want to run.


## DL parallel latency prediction
To predict the latency for a specific plan, the trained models are required. Run the full optimization first to get the trained models. Then open the file `predict_latency.py` and edit the parallel execution plans for which you want to predict the latency. Then use the following command to predict the latency.
```bash
time SAVED_MODELS_DIR='<path_to_saved_models>' python predict_latency.py --model=<model_name>
```