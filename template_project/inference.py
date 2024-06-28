import torch

def infer(model, image):
    model.eval()
    with torch.no_grad():
        output = model(image)
        prediction = output.round()
    return prediction.item


if __name__ == '__main__':
    main()