__Author__ = 'SARANG TAMRAKAR'
__Email__ = 'sarang.tamrakarsgi15@gmail.com'
__GitHub__ = 'sarangtamrakar'
__Version__ = '1.0.0'

try:
    import os
    import json
    import yaml
    import argparse
except Exception as e:
    raise e


class ProjectStructure:
    """
        class Name: ProjectStructure
        WrittenBy: SARANG TAMRAKAR
        Version: 1.0
        Description: This class specially designed for creating the Project Structure
    """

    def __init__(self):
        pass
    def make_dirs(self,lis_dirs):
        """
                        Method Name: make_dirs
                        WrittenBy: SARANG TAMRAKAR
                        Version: 1.0
                        Description: this method create the directorys for project structure
                        return: None
        """
        try:
            for dir in lis_dirs:
                os.makedirs(dir)
            print("all directories created")
        except Exception as e:
            raise e


if __name__ == '__main__':
    # list of directories & subdirectories
    dirs = ['src', 'notebooks', 'Data', 'Data/rawData', 'Data/ProcessedData','ModelData/CheckpointAUG','ModelData/UnetLogsAUG']
    projectObj = ProjectStructure()
    projectObj.make_dirs(dirs)
