from wide_resnet import wide_resnet



model = wide_resnet.WideResNet(image_size = 64,race = True, train_branch=True)()
model.summary()
