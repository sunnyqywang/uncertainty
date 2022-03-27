from torch.utils.data import Dataset


class CTA_Data(Dataset):

	def __init__(self, *args):
		self.data = []
		for i in range(len(args)):
			self.data.append(args[i])
		self.numItems = len(args)
		'''
		self.x = x
		self.y = y
		self.history = history
		self.qod = qod
		'''

	def __len__(self):
		return len(self.data[0])

	def __getitem__(self, idx):

		#sample = (self.x[idx],self.y[idx],self.history[idx],self.qod[idx])
		sample = tuple(self.data[i][idx] for i in range(self.numItems))

		return sample
