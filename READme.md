## Deep learning project

### Current model/architecture:
1. Resnet18
2. Resnet50

### How to run:
1. Prepare data in similar format in data directory.
2. Edit model in config.yml file
3. Run
``` shell 
python main.py
```
4. Run the following script to visualize training results (only train/valid loss)
``` shell 
tensorboard --logdir=path/to/logs/directory
```