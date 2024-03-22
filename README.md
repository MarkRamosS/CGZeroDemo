# CGZeroDemo
A simpler version of the bot that I used to play ultimate Tic-Tac-Toe. (https://www.codingame.com/multiplayer/bot-programming/tic-tac-toe)
Since this is an active competition and uploading my solution would ruin its purpose, so I am showcasing my implementation on the normal Tic-Tac-Toe rules.

AlphaZero is a bot that gets its strength from selfplay and the statistical nature of Monte Carlo Tree-Search (MCTS). In normal MCTS a rollout consists of the random play until a final state is reached and this value is returned. In AlphaZero, instead of the random rollout, this value is calculated from a simple DNN layer. I don't use the backpropagation part of the network that is used by AlphaZero, since its computation is too heavy for this kind of problem.  



The implementation consists on several parts:

A game simulator. It's important to have an accurate simulation, It must simulate a turn and give all the legal moves for the current player. Written in C++
A modified MCTS tree, that uses the NN as evaluation and exploration. Written in C++.

Different workers: Selfplay (play games and save states as samples), Pitplay (get winrates between generations), Sampler (prepare samples for training), submit (to send it to CG), written in C++.

A trainer. Written in Python, it uses Jupyter Notebook and uses Tensorflow as the ML framework.

A NN model. This is the definition of the Neural Network that will be used both in the C++ and the Python trainer. It has layers (Inputs, Dense Layers, and outputs).

In my implementation I have 5 main files:

CGZero.cpp: It does all the C++ roles except the Sampler.

NNSampler.cpp: Sampler Worker. Takes all samples, and average unique gamestates (it reduces samples size count). Right now is game agnostic. It takes inputs+outputs+count from a binary float file.

NN_Mokka.h: NN Inference engine.

Train.ipynb: Jupyter Notebook that does all the training.



