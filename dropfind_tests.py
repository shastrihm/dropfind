"""
dropfind_tests.py 

- Tests dropfind.py for expected behavior in different deployment scenarios
- Run this script in a seperate command prompt from dropfind

"""
import os, shutil
import time
import random 
import csv
import argparse 

WITHDRAW_DIR = "test_install"
DEPOSIT_DIR = "test_install/test"
DF_CATCHUP = 8 #number of seconds to wait for Ready for inference... from dropfind.py running concurrently 


class SimulateUser:
	def __init__(self, withdraw_dir, deposit_dir):
		self.from_dir = withdraw_dir # directory where images are located 
		self.to_dir = deposit_dir # directory to move images to 

	def one_shot_move(self, n = None):
		"""moves all images from withdraw_dir to deposit_dir in one go"""
		files = os.listdir(self.from_dir)
		i = 0
		for f in files:
			if f.endswith(".jpg"):
				shutil.move(self.from_dir + os.sep + f, self.to_dir + os.sep + f)
				i += 1
			if n is not None and i >= n:
				break
			
	def move(self):
		"""moves a single image from withdraw_dir to deposit_dir
		   Does nothing if no images left in withdraw_dir
		"""
		files = os.listdir(self.from_dir)
		for f in files:
			if f.endswith(".jpg"):
				shutil.move(self.from_dir + os.sep + f, self.to_dir + os.sep + f)
			break 

	def send_stop(self):
		"""saves stop.txt to deposit_dir""" 
		with open(self.to_dir + os.sep + "stop.txt", 'w') as f:
			f.write("stop")

	def reset(self):
		"""
		Resets system state from before the simulation.
			Moves all images from deposit_dir back to withdraw_dir
			Deletes the csv file and stop/exit text file acknowledgements
		"""
		for file in os.listdir(self.to_dir):
			if file != "testbed.txt":
				if file.endswith(".txt") or file.endswith(".csv"):
					os.remove(self.to_dir + os.sep + file)
				else:
					# file is a .jpg
					shutil.move(self.to_dir + os.sep + file, self.from_dir + os.sep + file)



def BasicTests(directory, correct_csv_rows):
	"""
	Tests basic properties in directory to check for dropfind script correctness
	"""
	# csv file named correctly 
	# csv file has correct number of rows 
	# dropfind_log was updated correctly 
	barcode = "J000597"
	files = os.listdir(directory)
	assert barcode + ".csv" in files, "csv file not detected or has incorrect filename"
	assert("exit.txt" in files)
	with open(directory + os.sep + barcode + ".csv") as f:
		reader = csv.reader(f)
		assert len(list(reader)) == correct_csv_rows,  directory + os.sep + barcode + ".csv has " + str(len(list(reader))) + " rows, should have " + str(correct_csv_rows)
	with open("dropfind_log.txt", "r") as f:
		num_lines = sum(1 for line in f)
	assert num_lines >= 1, "dropfind_log.txt did not update"
	assert num_lines <= 500, "dropfind_log.txt did not refresh"


def wait_for_exit(directory, timeout=60):
	"""
	Monitors directory for exit.txt. Stops monitoring after timeout seconds
	Returns True if exit.txt detected before timeout. Returns False if process times out
	"""
	i = 0
	interval = 2
	while i < timeout:
		if "exit.txt" in os.listdir(directory):
			return True
		i += interval
		time.sleep(interval)
	return False

def clear_dropfind_log():
	"""
	Clears log
	"""
	with open("dropfind_log.txt", 'w') as f:
		pass

########################################################
########################################################
########################################################
########################################################
########################################################

def test_install():
	"""
	Basic test to check if dropfind was installed successfully
	"""
	print("Running Test : Basic Functionality...")
	clear_dropfind_log()
	User = SimulateUser(WITHDRAW_DIR, DEPOSIT_DIR)
	User.reset()
	time.sleep(DF_CATCHUP)
	num_ims = len(os.listdir(WITHDRAW_DIR))-1
	User.one_shot_move()
	if wait_for_exit(DEPOSIT_DIR, timeout=60):
		BasicTests(DEPOSIT_DIR, correct_csv_rows = num_ims)
		print("Test Passed")
		print()
	else:
		User.reset()
		time.sleep(5)
		raise ValueError("Test failed. exit.txt not found by timeout")
		print()
		return False
	User.reset()
	return True


def test_longrun(interval=1.6):
	"""
	Test on a run of all images to completion.
	interval (float) : no. seconds to wait between putting images in deposit_dir
	"""
	print("Running Test : Complete Run, camera interval=" + str(interval) + "s...")
	clear_dropfind_log()
	User = SimulateUser(WITHDRAW_DIR, DEPOSIT_DIR)
	User.reset()
	time.sleep(DF_CATCHUP)
	num_ims = len(os.listdir(WITHDRAW_DIR))-1
	for i in range(num_ims):
		User.move()
		time.sleep(interval)
	if wait_for_exit(DEPOSIT_DIR, timeout=60):
		BasicTests(DEPOSIT_DIR, correct_csv_rows = num_ims)
		print("Test Passed")
		print()
	else:
		User.reset()
		time.sleep(5)
		raise ValueError("Test failed. exit.txt not found by timeout")
		print()
		return False
	User.reset()
	return True

def test_stop(interval=1.6):
	"""
	Test for correct behavior when prematurely terminated
	"""
	print("Running Test : Premature Stop, camera interval=" + str(interval) + "s...")
	clear_dropfind_log()
	User = SimulateUser(WITHDRAW_DIR, DEPOSIT_DIR)
	User.reset()
	time.sleep(DF_CATCHUP)
	i = 0
	stop_at = random.randint(0,len(os.listdir(WITHDRAW_DIR))-2)
	for i in range(len(os.listdir(WITHDRAW_DIR))):
		User.move()
		time.sleep(interval)
		if i == stop_at:
			User.send_stop()
			break 
		i += 1
	if wait_for_exit(DEPOSIT_DIR, timeout=60):
		with open("dropfind_log.txt") as f:
			if "stop.txt detected. Quitting prematurely..." not in f.read():
				User.reset()
				time.sleep(5)
				raise ValueError("dropfind did not detect stop.txt and did not quit prematurely")
				print()
				return False
			else:
				print("Test Passed")
				print()
	else:
		User.reset()
		time.sleep(5)
		raise ValueError("Test failed. exit.txt not found by timeout")
		return False

	User.reset()
	return True

def test_refreshlog():
	"""
	Test to see if dropfind_log.txt is correctly refreshed after 500 lines
	"""
	print("Running Test : dropfind_log rollover...")
	with open("dropfind_log.txt", 'w') as f:
		for i in range(501):
			f.write("dummy text\n")

	User = SimulateUser(WITHDRAW_DIR, DEPOSIT_DIR)
	User.reset()
	time.sleep(DF_CATCHUP)
	num_ims = len(os.listdir(WITHDRAW_DIR))-1
	User.one_shot_move()
	if wait_for_exit(DEPOSIT_DIR, timeout=60):
		BasicTests(DEPOSIT_DIR, correct_csv_rows=num_ims)
		print("Test Passed")
		print()
	else:
		User.reset()
		time.sleep(5)
		raise ValueError("Test failed. exit.txt not found by timeout")
		print()
		return False
	User.reset()
	return True



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-t", "--test", help="Test to run (str). One of 'basic', 'longrun', 'stop' or 'rollover'", type=str)
	parser.add_argument("-i", "--interval", default=1, help="Period of time to wait between dumping images in test directory", type=float)
	args = parser.parse_args()
	test = args.test 
	interval = args.interval
	if test == "basic":
		test_install()
	elif test == "longrun":
		test_longrun(interval=interval)
	elif test == "stop":
		test_stop(interval=interval)
	elif test == "rollover":
		test_refreshlog()
