# GAMMAF: A Common Framework for Graph-Based Anomaly Monitoring Benchmarking on LLM-based Multi-Agent Systems

## Overview

We introduce **GAMMAF**, an open-source framework for the generation of communication data in LLM-based multi-agent systems (LLM-MAS) and the evaluation of topology guided defense methods against attacks to the integrity of the system. **GAMMAF** is not a novel defense mechanism itself, but rather a comprehensive evaluation architecture designed to generate synthetic multi-agent interaction datasets and benchmark the performance of existing and future defense models. The proposed framework operates through two interdependent pipelines: a Training Data Generation stage, which simulates debates across varied network topologies to capture interactions as robust attributed graphs, and a Defense System Benchmarking stage, which actively evaluates defense models by dynamically isolating flagged adversarial nodes during live inference rounds.

<img width="100%" alt="functionDiagram_more" src="https://github.com/user-attachments/assets/23b3b611-1c28-4f5d-ba85-04b186d78afb" />

## Quick start

First create a python environment and install all the necessary dependencies:

```
conda create -n gammaf-env python=3.11
conda activate gammaf-env
pip install -r requirements.txt
```

Next step is to setup the backbone LLM variables. Update ```.env``` with the corresponding BASE_URL, API_KEY and MODEL_NAME (see [.env.example](.env.example) for the required format). If you were using a local inference service that does not require an api key, still include this field with a placeholder or there will be errors during runtime.

Now we are ready to generate debate training data. We set the config using [generation config template](config-examples) and run with
```
# Generate debate training data using the provided template
python TrainDataGeneration.py config-examples/generation-config.yaml
```

Once the training data is generated, we can go to the defense model evaluation stage, using as config [evaluation config template](config-examples).
```
# Run evaluation stage for all loaded defense architectures
python MainEvaluation.py config-examples/evaluation-config.yaml
```

Next we offer a comprehensive guide on how to modify the architecture for specific tests: adding new defense models, adding new text output processing logic and adding new tasks datasets.

## Customized Benchmarking
### Adding new defense architectures
### Adding new model output processing
### Adding new tasks datasets
