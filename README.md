# Sequence Generation using a Character-level LSTM Model trained on Tolstoy's Anna Karenina

## Table of Contents
* [Corpus](#corpus)
* [Requirements](#requirements)
* [Architecture](#architecture)
* [Setup](#setup)
* [Hyperparameters](#hyperparameters)
* [Sample Generation](#sample-generation)

## Corpus
Anna Karenina is a novel by the Russian author Leo Tolstoy, first published in book form in 1878. Many writers consider Anna Karenina the greatest work of literature ever, and Tolstoy himself called it his first true novel.
This book is available for free in public domain.

## Requirements
Project is created with:
* Python: 3.7.4
* PyTorch: 1.7.1
* torchvision: 0.8.2

## Architecture
Two stacks of LSTM layers are stacked on one another with a dense layer connected at the output with a sigmoid activation. The output is one time step shifted in the future for character-level text generation.

## Setup
### To train the model, run this code in the directory
`python train.py`
### To generate sample
`python generate.py`

## Hyperparameters
epochs = 20<br />
learning_rate = 0.001<br />
clip_gradient = 5<br />
validation_fraction = 0.1<br />
n_hidden = 512<br />
batch_size = 128<br />
seq_length = 100<br />
n_layers = 2<br />

## Sample Generation
<strong>For input text : He said</strong><br /><br />
that she was too walked to her friend on a contrary of her face to her                  
tanting on his hand with a sund of stopped this compane, and a candle                   
and tears standing the colonel, seemed to him.                                          
                                                                                        
"What do you share all that!"                                                           
                                                                                        
"Not it sawished."                                                                      
                                                                                        
"Oh, no!" the place there were attaching this point of talking                          
of his brows.                                                                           
                                                                                        
"All at him? We'll be in all made that is to be answered. Well, and I'll                
stand an instant."                                                                      
                                                                                        
"You want for a little being that have said an instant, and the doctor's                
worss, and have threonger for a mastice the sight. And his whole conversation and       
suffering in the signs of the freedom to himself to stand on the                        
pool of those which she does not answer, what are you through it, but it's              
all meant. I went to had a chance of the same."                                         
                                                                                        
"When to see the second thing."                                                         
                                                                                        
The same position the conscalitude of the face of his face shoted the previous          
pensive on to this coat too, and the face in the path of her beath of                   
the force over her eyes; that                                                           
howering themselves, she came in.                                                       
                                                                                        
"You did not ask her it all over it? What a contemptoom of the supper answer            
is to speak of the most point for the chiment, and so to do. Here                       
are not to be made. If it will say.... The countess would share and the sound           
of the constretsing in some other to do at happiness, would be                          
docting that her meaning of the commessore as though the princess,                      
they must, but as to taking the mistake of her hand of the                              
conversation with him to give up himself intended to be in this. And is that this       
were their strange, but she's not married. It's not time of it. A thick of              
myself talking to me. And I have been so much to bele at the fact                       
in, a political course in the candicaty of mind, to get the contrary,                   
and how she has been true, and share her hands of the condition of                      
the conversation with the comprohing of the foreses to the soul to seat the             
personary--while a servant council was as that there was not such a men. I went         
together them women to be an intellent of troubles without his brother's                
