
# Deployment with `TorchServe`

## Objetive
_______
Taxonomy classifier deployment with `TorchServe`.

`TorchServe` is a simple and easy-to-use tool developed by Facebook and AWS to deploy `Pytorch` models in production.

## Model
________
The model weigths can be downloade at the following link:

https://drive.google.com/drive/folders/1LOnXoH2iCOBtD9of17uqa4PTFTQ2h9eX?usp=sharing

## Deployment
_________

``````
# create .mar file
torch-model-archiver --f \
--model-name taxonomyClassifier \
--version 2.0 \
--model-file model.py \
--serialized-file taxonomy_approach_1_v5.bin \
--extra-files index_to_name.json \
--handler model_handler.py

# start torchserve
torchserve --start --model-store model_store

# register model
curl -X POST "http://localhost:8081/models?url=taxonomyClassifier.mar"

# scale workers
curl -v -X PUT "http://localhost:8081/models/taxonomyClassifier?min_worker=4"

``````

## Results
_________
- Inference:
````
# inference
curl http://localhost:8080/predictions/taxonomyClassifier -T sample_text.txt
➜ Taxonomy 17
````

- Logs during inference
````
2021-06-24 21:15:37,630 [INFO ] W-9000-taxonomyClassifier_2.0-stdout MODEL_LOG - Esto es un ejemplo de prueba
2021-06-24 21:15:37,631 [INFO ] W-9000-taxonomyClassifier_2.0-stdout MODEL_LOG - Received texts: ['Esto es un ejemplo de prueba']
2021-06-24 21:15:37,631 [INFO ] W-9000-taxonomyClassifier_2.0-stdout MODEL_LOG - Input text type: <class 'list'>
2021-06-24 21:15:37,631 [INFO ] W-9000-taxonomyClassifier_2.0-stdout MODEL_LOG - Number of examples: 1
2021-06-24 21:15:48,499 [INFO ] W-9000-taxonomyClassifier_2.0-stdout MODEL_LOG - This the output from the Multiclass
classification model [0.0053795091807842255, 0.00020787259563803673, 0.011930802837014198, 0.0023118683602660894,
0.8826913833618164, 0.06656178086996078, 0.009695082902908325, 0.02122172713279724]
2021-06-24 21:15:48,499 [INFO ] W-9000-taxonomyClassifier_2.0-stdout MODEL_LOG - ['Taxonomy 173']
````

## Environment
_______
The deployment was done in a local environment. However, the process is valid for deployment in a cloud instance such as an AWS EC2
instance.

Two virtual environments were created for the project utilizing the anaconda distribution, one for development and one for deployment with `TorchServe`.
To reproduce the results, the virtual environment can be installed from the following lines of code:

- macOS:

`$ conda create --name torchserve --file requirements_torchserve.txt`

- Windows o Linux:

`$ conda env create -f environment_torchserve.yml`