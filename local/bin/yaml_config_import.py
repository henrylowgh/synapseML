import yaml

# Load configurations from YAML file
with open("config.yaml", 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Accessing configurations in the program
image_resize = tuple(config['data']['image_resize'])
batch_size = config['training']['batch_size']
epochs = config['training']['epochs']
base_model_choice = config['model']['base_model']
