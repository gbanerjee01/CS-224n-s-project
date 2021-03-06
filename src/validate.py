import torch
import utils

def evaluate(model, device, test_loader, loss_fn):
	correct = 0
	total = 0
	loss_avg = utils.RunningAverage()

	model.eval()
	with torch.no_grad():
		for batch_idx, data in enumerate(test_loader):
			inputs = data[0].to(device)
			target = data[1].squeeze(1).to(device)

			outputs = model(inputs)

			_, predicted = torch.max(outputs.data, 1)
			total += target.size(0)
			correct += (predicted == target).sum().item()

			loss = loss_fn(outputs, target - 1)
			loss_avg.update(loss.item())

	return loss_avg(), (100*correct/total)