from torchvision import models
from torch.functional import F
import torch


model = models.resnet101(pretrained=True)
script_model = torch.jit.script(model)
script_model.save('./model_store/deployable_model.pt')
model_scripted = torch.jit.script(model)


dummy_input = torch.rand(1, 3, 224, 224)
k=5
unscripted_output = model(dummy_input)
scripted_output = model_scripted(dummy_input)

unscripted_top5 = F.softmax(unscripted_output, dim=1).topk(5).indices
scripted_top5 = F.softmax(scripted_output, dim=1).topk(5).indices

print('Python model top 5 results:\n  {}'.format(unscripted_top5))
print('TorchScript model top 5 results:\n  {}'.format(scripted_top5))