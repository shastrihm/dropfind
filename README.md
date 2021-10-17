# Object detection for crystallization drops
A deep learning model to detect the center of drops trained with the Tensorflow2 object detection API. Special purpose inference script provided. 
## Requirements
- Windows OS 
- Python 3.6 (may work with newer Python versions, but this has not been tested)
- Internet connection
## How to install
1) Download and install Python 3.6. There are many ways to do this. The most straightforward way may be to download the installer by clicking [here](https://www.python.org/ftp/python/3.6.2/python-3.6.2-amd64.exe). Open it and follow the on screen instructions until it says the installation is successfull. 
    - After installing, running `python` in the command line may not work. If this is the case, you may need to add Python to your environment variables. To do this, rerun the installer and select **Modify -> Next**, then check the box that says "Add Python to environment variables", and then click Install.
2) Download and install Microsoft Visual Studio [here](https://c2rsetup.officeapps.live.com/c2r/downloadVS.aspx?sku=community&channel=release&source=vslandingpage&cid=2011) if you don't already have a copy installed. When following the installer instructions, select "Desktop development with C++" as a workload to install. 
3) Clone this repository to your machine, either via `git clone https://github.com/shastrihm/dropfind.git`  in Git Bash or by downloading and unzipping the zip file. 
4) In the command line, from the directory that you cloned it to, run `setup.bat`. This should install all the required dependencies for running the inference script.

## Testing your installation

Run `test.bat` from the directory you cloned the repo to and wait for it to complete. If the installation was successful, no errors will be thrown and the message 

`Test passed. Installation successful.` 

will be printed to the console. 

## Usage 
Note : Tensorflow GPU warnings are suppressed. Unsuppress them by commenting out the `logging.getLogger('tensorflow').setLevel(logging.FATAL)` at the beggining of `dropfind.py`.

The `dropfind.py` script takes the following command line arguments
1) `-p` (str): The absolute or relative path to the **empty** directory that will be populated with images (wrap this in quotes if your path contains a space. If applicable, make sure to use `\\` in place of `\` to avoid escape character shenanigans)
2) `-n` (int): The number of images that will populate the directory
3) `-b` (str): The barcode of the batch of images that will populate the directory. This dictates the resulting filename of the output csv file.
4) `-m` (bool): Whether or not to suppress all console output. Defaults to True if not specified.
5) `-t` (bool): Whether or not to operate in test mode. Defaults to False. Should only be true in `test.bat`.

E.g. `python dropfind.py -p "C:\Path\To\Images" -n 42 -b "J000597" -m False`
- The script will begin monitoring the directory `C:\Path\To\Images`. Once it prints `Ready for Inference...` to the console (which it will not print if `-m` is False), it will perform inference on each image as it enters the directory, writing data to a `temp.csv` file. After `42` images have been processed, the script will terminate. It will rename the `temp.csv` to `J000597.csv` in the same directory, with the drop center coords and filenames for each processed file: 

`C:\Path\To\Images\J000597.csv`
     
| image1.jpg | (x,y) coordinates of inferred drop center for image1 |
|------------|------------------------------------------------------|
| . . .      | . . .                                                |
| image42.jpg | (x,y) coordinates of inferred drop center for image42 |      
          



