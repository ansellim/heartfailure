# This is the minimal amount of code required to execute prototype2.py code on Google Colab or any cloud server.

import os
from getpass import getpass
import urllib.parse
import shutil

user = input('User name: ') # Enter georgia tech username
password = getpass('Password: ') # Enter password
password = urllib.parse.quote(password) # your password is converted into url format

cmd_string = 'git clone https://{0}:{1}@github.gatech.edu/jlee3702/CSE6250_Project.git'.format(user, password)

os.system(cmd_string)
cmd_string, password = "", "" # removing the password from the variable
del password
del user
del cmd_string

os.chdir("./CSE6250_Project/neural/")

