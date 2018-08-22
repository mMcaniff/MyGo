import config

def log(message):
   if config.DEBUG:
      print("DEBUG " + message) 
   
