
TrainEmbMultiTask
======
TrainEmbMultiTask is a package for training character representation with multitask training. It is proposed for enhancing Chinese Word Segmentation task in our ACL 2017 paper [Neural word segmentation with rich pretraining](insert paper link), while it can be extended to other task which need to build a better sub-unit representation (e.g. character representation for word segmentation; word representation for entity recognition.) through multitask training. Currently the code support training in four different tasks. The output `.pmodel, .pchar, .pbichar` can be loaded by [RichWordSegmentor](https://github.com/jiesutd/RichWordSegmentor) to initialize the character representation parameters and input character embedding and character bigram embedding, respectively.

Structure:
=====
It is a simple two-layer feed forward neural networks which support multitask training. All input task instances are randomly choice during training (user can configure the instance ratio of different tasks.).


Installation:
======
* Download the [LibN3L](https://github.com/SUTDNLP/LibN3L) library and configure your system. Please refer to [Here](https://github.com/SUTDNLP/LibN3L)
* Open [CMakeLists.txt](CMakeLists.txt) and change " ../LibN3L/" into the directory of your [LibN3L](https://github.com/SUTDNLP/LibN3L) package.


`cmake .`  

`make JointTrainemb`

Training character representation: 
`./JointTrainemb -l -atrain {a_train_file} -adev {a_dev_file} -btrain {b_train_file} -bdev {b_dev_file} -ctrain {c_train_file} -cdev {c_dev_file} -dtrain {d_train_file} -ddev {d_dev_file} -char {pretrained_char_embedding} -bichar {pretrained_char_bigram_embedding} -option {option_file} -model {save_to_model_file} -pmodel {save_to_representation_file}`

Parameters:
`-atrain; -btrain; -ctrain; -dtrain`: training data for four task;
`-adev; -bdev; -cdev; -ddev`: development data for four task;
`-char`: pretrained character embeddding;
`-bichar`: pretrained charcacter bigram embedding;
`-option`: configure parameters on building the training structure, like the set iteration times, embedding dimension, .etc.
`-model`: the directory for saving whole model;
`-pmodel`: the directory for saving the character representation. 


Input:
======
Each line contains one token and its label. A null line seperate two sentences.
Take segmentation as example:  

国 b-seg
务 m-seg
委 m-seg
员 e-seg
兼 s-seg
国 b-seg
务 m-seg
院 e-seg
秘 b-seg
书 m-seg
长 e-seg
罗 s-seg
干 s-seg
、 s-seg

Output:
=======
Our code may generate may outputs, like the decoded result of development data, the whole model and the character representation `pmodel` and character with character bigram embeddings `.pchar, .pbichar`.   
The `.pmodel` file contains parameters (weights) of input character representation network.
The `.pchar` file contains character embeddings after the multitask training.
The `.pbichar` file contains character bigram embeddings after the multitask training.  
(We suggest you combine the `.pchar/.pbichar` embeddings with the initial input pretrained char/bichar embeddings together, i.e. if char/bichar exists in the multitask output embedding, then keep the multitask output embedding; else keep the initial input pretrain embedding. This can effectively reduce OOV and get a better performance in following tasks.)


Cite:
=====


Note: 
======
* Current version only compatible with [LibN3L](https://github.com/SUTDNLP/LibN3L) after ***Dec. 10th 2015*** , which contains the model saving and loading module.

