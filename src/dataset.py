import torch
import torchvision
import matplotlib.pyplot as plt
from typing import Any
from torchvision.utils import _Image_fromarray



n_epochs = 3
trainBatchSize = 64
testBatchSize = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
# torch.manual_seed(random_seed)



class PairedMNIST(torchvision.datasets.MNIST):
    def __getitem__(self, index: int) -> tuple[tuple[Any, Any], tuple[Any,Any], int]:

        img1, target1 = self.data[index], int(self.targets[index])
        img2, target2 = self.data[(index + 1) % len(self.data)], int(self.targets[(index + 1) % len(self.data)])
        
        img1 = _Image_fromarray(img1.numpy(), mode="L")
        img2 = _Image_fromarray(img2.numpy(), mode="L")

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if self.target_transform is not None:
            target1 = self.target_transform(target1)
            target2 = self.target_transform(target2)
        
        parity = (target1 + target2) % 2

        return (img1, img2), (target1, target2), parity



train_loader = torch.utils.data.DataLoader(PairedMNIST(
    root="./dset", 
    train=True,
    download=True,
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=trainBatchSize, shuffle=True)

test_loader = torch.utils.data.DataLoader(PairedMNIST(
    root="./dset",
    train=False,
    download=True,
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=testBatchSize, shuffle=True)

images, labels, parities = next(iter(train_loader))
image1s, image2s = images
label1s, label2s = labels
print(parities)

cols, rows = 2, 3
figure = plt.figure()
for i in range(1, 7):
    sample_idx = torch.randint(len(image1s), size=(1,)).item()
    (img1, img2), (label1, label2) = (image1s[sample_idx], image2s[sample_idx]), (label1s[sample_idx], label2s[sample_idx])
    parity = parities[sample_idx]
    stitched_img = torch.cat((img1, img2), dim=2) 
    figure.add_subplot(rows, cols, i)
    plt.axis("off")
    plt.title(f'Pair: ({label1},{label2}) | Parity: {parity}')
    plt.imshow(stitched_img.squeeze())
plt.show()

    
    


