VERBOSITY = 1
SILENT = 0
MODERATE = 1
DEBUG = 2

def debugPrint(string, thresh):
  if (thresh <= VERBOSITY):
    print(string)