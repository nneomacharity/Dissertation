
#CREATE NEW ENVIRONMENT IF YOU HAVE NOT PREVIOUSLY
python -m venv myenv


CHANGE TO NEW ENVIRONMENT
FOR WINDOWS BASH
source myenv/bin/activate

FOR LINUX AND MAC BASH
source myenv/bin/activate




## TO HANDLE PACKAGES FOR INSTALLATION ON ANY ENVIRONMENT

TO FREEZE --- BUT YOU DON'T NEED TO I HAVE DONE THIS ALREADY.
To freeze packages e.g [pip] [freeze] > [name-of-file.txt]
pip freeze > install_packages.txt


JUST INSTALL WITH THIS
To install freezed packages e.g [pip] [install] -r [name-of-file.txt]
pip install -r install_packages.txt
