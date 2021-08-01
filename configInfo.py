__Author__ = 'SARANG TAMRAKAR'
__Email__ = 'sarang.tamrakarsgi15@gmail.com'
__GitHub__ = 'sarangtamrakar'
__Version__ = '1.0.0'

try:
    import yaml
except Exception as e:
    raise e


def readConfig():
    """
            Method Name: readConfig
            WrittenBy: SARANG TAMRAKAR
            Version: 1.0
            Description: This method Read the configuration from params.yaml file.
            return: configdata

    """
    with open('params.yaml','r') as f:
        configFile = yaml.safe_load(f)
    return configFile
