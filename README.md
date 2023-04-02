# Turkish to Turkish Sign Language (TSL) Translation
This repository provides several translation models that is used for translating Turkish texts to TSL equivalents. Neural Machine Translation (NMT) methods used to achieve this. There are several networks used in the project:
* Recurrent Neural Network (RNN) based architectures
  * LSTM
  * GRU
  * BLSTM
  * BGRU
* Transformers

## Usage
### Training
* Using the model configuration file, the training setup can be started easily

    python src\train.py --model-config configs\networks\transformer.py --batch-size 16 --token-type word --num-epochs 20 

### Testing
* The "--exp-path" parameter is the root path of the experiment. By default, all the experiments are saved under the "results" folder in the root path of the project. You need to provide the name of model model used and the id of your experiment as in the below input. 

    python src\test.py --model-config configs\networks\transformer.py --token-type word --exp-path results\transformer\<exp-id>
    
The results obtained from this project are gathered in <this-paper>.
.
.
.
