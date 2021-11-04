# Object detection for crystallization drops
Real time inference for a drop center detecting deep learning model trained with the Tensorflow2 object detection API. 
Made for LabVIEW integration on Windows. 

## Requirements
- Windows OS 
- Python 3.6 (may work with newer Python versions, but this has not been tested)
- Internet connection
- Git Bash (for git commands. Not necessary but makes things a lot easier.)

## How to install
1) Download and install Python 3.6. There are many ways to do this. The most straightforward way may be to download the installer by clicking [here](https://www.python.org/ftp/python/3.6.2/python-3.6.2-amd64.exe). Open it and follow the on screen instructions until it says the installation is successfull. 
    - After installing, running `python` in the command line may not work. If this is the case, you may need to add Python to your environment variables. To do this, rerun the installer and select **Modify -> Next**, then check the box that says "Add Python to environment variables", and then click Install.
2) Download and install Microsoft Visual Studio [here](https://c2rsetup.officeapps.live.com/c2r/downloadVS.aspx?sku=community&channel=release&source=vslandingpage&cid=2011) if you don't already have a copy installed. When following the installer instructions, select "Desktop development with C++" as a workload to install. 
3) Clone this repository to your machine by running `git clone https://github.com/shastrihm/dropfind.git`  from your desired directory in Git Bash.
4) In the command line, from the directory that you cloned it to, run `setup.bat`. This should install all the required dependencies for running the inference script.

## Testing your installation
1) Run `test_imports.bat` to check if the required dependencies can be imported without error. If so, you will see a message `Imports successful` printed to the console.
2) Run `test.bat` to test functionality. It may take a few minutes. If the installation was successful, no errors will be thrown and the message `Test passed` will be printed to the console for each test. As well as testing basic installation, the testing suite (implemented in `dropfind_tests.py`) verifies correct behavior from dropfind in various deployment scenarios.

## Updating
If this GitHub repository is more up-to-date than your local repo, and you would like to update it: from the directory where you cloned this repo to, run the command `sh update.sh` in **Git Bash** (not Windows command line, unless you have taken special steps to ensure you can run shell scripts from the windows command line). 
- Warning: Because updating will force your local repo to become identical to the remote (github) repo, this will overwrite any changes or additions you made locally. So make sure to backup your local repo if you want to preserve them.

## Usage 
Note : Tensorflow GPU warnings are suppressed. Unsuppress them by commenting out the `logging.getLogger('tensorflow').setLevel(logging.FATAL)` at the beggining of `dropfind.py`.

The `dropfind.py` script takes the following command line arguments
1) `-p` (str): The absolute or relative path to the empty directory that will be populated with images (wrap this in double quotes if your path contains a spaces or special characters)
2) `-n` (int): The number of images that are expected to populate the directory
3) `-b` (str): The barcode of the batch of images that will populate the directory. This dictates the resulting filename of the output csv file.
4) `-m` (bool): Whether or not to suppress all console output to the console. Defaults to True if not specified. Note that console output will be directed to `dropfind_log.txt` regardless.

E.g. `python dropfind.py -p "C:\Path\To\Images" -n 42 -b "J000597" -m False`
- The script will begin monitoring the directory `C:\Path\To\Images`. If this directory has not been created yet, the script will enter a loop to check every few seconds whether it has been created, and once it has, will proceed normally. The script will timeout after 60 seconds if the directory still does not exist by then.
- Once it prints `Ready for Inference...` to the console (you can check `dropfind_log.txt` for this if `-m` is True, see below), it will perform inference on images as they enter the directory, writing data to a `temp.csv` file. 
- After `42` images have been processed, the script will terminate, acknowledging its termination by saving a file named `exit.txt` in the directory. It will rename the `temp.csv` to `J000597.csv` in the same directory, with the drop center coordinates and filenames for each processed file: 

`C:\Path\To\Images\J000597.csv`
     
| image1.jpg | (x,y) coordinates of inferred drop center for image1 |
|------------|------------------------------------------------------|
| . . .      | . . .                                                |
| image42.jpg | (x,y) coordinates of inferred drop center for image42 |      


### Quitting dropfind prematurely
There may be cases where you want to terminate the program prematurely (i.e. before `-n` images have been processed) and cannot interrupt via the command line. To solve this, the script will monitor the directory specified by `-p` for a file named `stop.txt`. If `stop.txt` is detected, dropfind will terminate even if it is expecting more images. An acknowledgement `exit.txt` will be saved to the same directory so the user can verify that the script terminated. Note that directory is not deleted, so that must be handled by the user. 

### Logging
To facilitate debugging when console output is muted, dropfind will write logs to `dropfind_log.txt` within the directory containing the script (`dropfind.py`). 
Each time the script is executed, it will write script arguments, images it has performed inference on, and other information to `dropfind_log.txt`.
Each script execution will be demarcated by a line of dashes within the log. To manage space, if `dropfind_log.txt`exceeds 500 lines before starting a new script execution, it will erase its contents and start over. 

In the case of exceptions raised during script execution, dropfind will write the exception and metadata to both `dropfind_log.txt` and `dropfind_errors.txt`. Unlike `dropfind_log.txt`, `dropfind_errors.txt` will not be automatically refreshed.



