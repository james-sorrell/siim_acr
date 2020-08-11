import os

VERBOSITY = 1
SILENT = 0
MODERATE = 1
DEBUG = 2

def debugPrint(string, thresh):
  if (thresh <= VERBOSITY):
    print(string)

def createFolder(path):
  try:
      # Create target Directory
      os.mkdir(path)
      debugPrint("Directory {} created.".format(path), 1)
  except FileExistsError:
      debugPrint("Directory {} already exists.".format(path), 1)