import torch
import numpy as np

def individual_prediction(y_prev,xm,xs,w,l,r,g,bm,bs):
	# gives predictions for an indiviual sample 
	
	S,T = y_prev.shape
	
	if len(xm.shape) == 1:
		xm = xm.view(-1,1)
	assert xm.shape[1] == len(bm)-1
	if len(xs.shape) == 1:
		xs = xs.view(-1,1)
	assert xs.shape[1] == len(bs)-1

	#temp = torch.cholesky_inverse(torch.eye(S)-r*w)
	y_fit = bm[0]+ torch.matmul(xm,bm[1:].view(-1,1))
	y_fit += torch.matmul(y_prev.view(S,T), l.view(-1,1))
	y_fit += torch.matmul(torch.matmul(w, y_prev), g.view(-1,1))
	y_fit = torch.matmul(torch.eye(S)+r*w, y_fit)
	#y_fit = torch.squeeze(torch.matmul(temp, y_fit))
	
	std = bs[0]+torch.matmul(xs,bs[1:].view(-1,1))
	std = torch.matmul(torch.eye(S)+r*w, std)
	#std = torch.abs(torch.diag(temp)) * std
	
	return torch.squeeze(y_fit), torch.squeeze(std)

def individual_ll(y,xm,xs,w,l,r,g,bm,bs,dist='normal'):
	# calcuates likelihood for an individual sample

	# y: spatial units x time lag (SxT)
	# x: spatial units x exog variables (SxK)
	# l,r,g,b: stand for lambda, rho, gamma, beta (K)
	# dist: string, the assumed distribution of y
	
	S,T = y.shape

	assert len(l) == T-1
	if len(xm.shape) == 1:
		xm = xm.view(-1,1)
	assert xm.shape[-1] == len(bm)-1
	if len(xs.shape) == 1:
		xs = xs.view(-1, 1)
	assert xs.shape[-1] == len(bs)-1
	assert w.shape[0] == w.shape[1] == S
	
	y_fit, std = individual_prediction(y[:,:-1],xm,xs,w,l,r,g,bm,bs)
	
	'''
	if dist == 'lognorm':
		# the underlying normal distribution has mean and std: y_fit and std
		ll += scipy.stats.lognorm.logpdf(y, s=std, scale=y_fit)
	'''
	
	if dist == 'normal':
		ll = -torch.sum(torch.distributions.normal.Normal(y_fit, std).log_prob(y[:,-1]))
		#print(scipy.stats.norm.logpdf(y, loc=y_fit, scale=std))
		
	return ll


def likelihood(params, *args):
	# calculates likelihood for the samples given
 
	(y,y_ts,xm,xs,w,lookback,predict_hzn,dist)=args
	
	l = params[0]
	r = params[1]
	g = params[2]
	bm = params[3]
	bs = params[4]
	
	ll = 0
	valid_samples = 0
	
	for i in y_ts:
		invalid = False
		for j in range(0, lookback):
			if (y_ts==(i-predict_hzn-j)).nonzero().nelement() == 0:
				invalid=True
				break
			
		idx = (i==y_ts).nonzero()[0]
		if not invalid:
			valid_samples += 1
			y_temp = torch.transpose(torch.cat((y[idx-lookback-predict_hzn+1:idx-predict_hzn+1], y[idx])),0,1)
			ll += individual_ll(y_temp,
								torch.squeeze(xm[idx]),
								torch.squeeze(xs[idx]),
								w,l,r,g,bm,bs,dist='normal')
	
	ll = ll/valid_samples
	ll = ll/y.shape[1]
	
	#print("%.2e" % (ll), end='\t')
	
	return ll


def predict(params, *args):
	# calculates predicted mean and standard deviations for all samples given 
	
	(y,y_ts,xm,xs,w,lookback,predict_hzn,dist)=args
	
	l = params[0]
	r = params[1]
	g = params[2]
	bm = params[3]
	bs = params[4]
	
	ll = 0
	valid_samples = 0
	y_fit_all = None
	std_all = None
	y_all = None
	
	for i in y_ts:
		invalid = False
		for j in range(0, lookback):
			if (y_ts==(i-predict_hzn-j)).nonzero().nelement() == 0:
				invalid=True
				break
			
		idx = (i==y_ts).nonzero()[0]
		if not invalid:
			valid_samples += 1
			y_fit, std = individual_prediction(torch.transpose(y[idx-lookback-predict_hzn+1:idx-predict_hzn+1],0,1),
								torch.squeeze(xm[idx]),torch.squeeze(xs[idx]),w,l,r,g,bm,bs)
			if y_fit_all is None:
				y_fit_all = y_fit
				y_all = torch.squeeze(y[idx])
				std_all = std
			else:
				y_fit_all = torch.cat((y_fit_all, y_fit))
				y_all = torch.cat((y_all, torch.squeeze(y[idx])))
				std_all = torch.cat((std_all, std))
	return y_all.detach().numpy(), y_fit_all.detach().numpy(), std_all.detach().numpy()
