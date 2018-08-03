# Landing rockets

Tensorflow code based on https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/10_A3C

To start training from scratch, create a new directory `saves/<mydir>` and a configuration file `saves/<mydir>/config.py` (see existing models in `saves`). Training is started by calling `python3 run.py <mydir>`.

Available flags are 

- `-t, --test`: Run in test mode. In this mode no training is performed and visualization is enabled. 
- `-n, --n_episodes`: Train for a fixed number of episodes (useful when submitting jobs to a server). Without this flag, the program will run indefinitely until user interrupt. 

Run `tensorboard --logdir=saves/<mydir>/log` to start tensorboard and monitor training progress on `http://localhost:6006/`

![Rocket landing](landing.gif)
