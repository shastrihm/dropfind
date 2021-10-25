"""
dir_watcher.py 

A class for watching directories for new .jpg files
"""
import os


class DirWatcher:
	def __init__(self, directory):
		# directory (str) : directory path
		self.dir = directory
		self.files = []

	def refresh_dir(self):
		"""
		updates list of files in directory.
		Returns list of .jpg files that have been newly detected (i.e. files that were not seen previously but now exist in the directory)
		"""
		new = os.listdir(self.dir)
		old = self.files 
		ret = list(filter(lambda f : f.endswith('.jpg'), list(set(new) - set(old))))
		self.files = new
		return ret

	def is_stop(self):
		return "stop.txt" in self.files

	def is_exit(self):
		return "exit.txt" in self.files

	def count(self):
		"""
		Returns number of .jpg images in directory
		"""
		return len([f for f in self.files if f.endswith('.jpg')])
 