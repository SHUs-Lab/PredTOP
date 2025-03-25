### PredTOP

This repository implements PredTOP, an latency predictor for distributed deep learning training with hybrid parallelism. PredTOP implements a latency predictor based on Transformers over Directed Acyclic Graphs(DAGs). Its primary objective is to aid automatic DL optimizers to quickly formulate the optimal parallel DL training plan without running large amount of profiling tasks to measure the runtime information of DL stages and operators.


#### Installation

1. PredTOP is based on the [Alpa](https://https://github.com/alpa-projects/alpa) compiler for DL training. Before beginning, install Alpa following the [instructions](https://alpa.ai/install.html).

2. Clone the source code of PredTOP from this repository.
    ```bash
    git clone https://github.com/SHUs-Lab/PredTOP.git
    ```

3. Install PredTOP in your python environment
    ```bash
    cd predtop
    pip install -e .
    ```

4. See the `Benchmarks` folder for examples on running the optimizer with PredTOP and predicting the latency.

