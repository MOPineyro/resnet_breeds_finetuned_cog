from typing import Any
from cog import BasePredictor, File, Input, BaseModel
import torch
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet
from PIL import Image
import numpy as np
import requests, io, logging, json
import torch.nn as nn

class Output(BaseModel):
    breed: str

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # Load the class names
        with open('class_names.json', 'r') as f:
            self.class_names = json.load(f)
        self.model = self.model_init()
    
    def model_init(self) -> ResNet:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        model = resnet18(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 37)

        model.load_state_dict(torch.load('image_classification.pth', map_location=device))  # Load the state dictionary
        model.to(device)  # Ensure the model parameters are on the correct device
        model.eval()

        return model
    
    def predict(self, image_file: File = Input(description="Upload an image file. (optional)", default=None),
                      image_url: str = Input(description="Enter an image URL. (optional)", default=None)) -> Output:
        """Run prediction on an image provided either as a file upload or via a URL."""
        if image_file is not None:
            image = Image.open(image_file)
        elif image_url is not None:
            image = self.image_get(image_url)
        else:
            raise ValueError("No image provided. Please upload an image file or enter an image URL.")
        
        if image is None:
            raise ValueError("Image URL error.")
        
        image = self.image_transform(image)
        
        # Run the prediction
        output = self.model(image)
        
        # Get the class with the highest score
        _, pred = torch.max(output, 1)
        
        # Return the breed name
        breed = self.class_names[pred.item()]
        
        return Output(breed=breed)


    def image_get(self, image_url: str) -> Image:
        try:
            logging.info(f"Retrieving image from: {image_url}")
            response = requests.get(image_url)
            response.raise_for_status()  # Raise an exception if the GET request was unsuccessful
        except requests.exceptions.RequestException as err:
            logging.error(f"Error occurred: {err}")
            return None

        image = Image.open(io.BytesIO(response.content))
        logging.info(f"Successfully retrieved image from {image_url}")
        return image

    def image_transform(self, image: Image) -> torch.Tensor:
        transform_resnet = transforms.Compose(
            [
                LetterboxImage(224), # Resnet-specific image size
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ],  # Resnet-specific standardization values
                ),
            ]
        )
        image = transform_resnet(image)
        image = image.unsqueeze(0)
        return image

class LetterboxImage:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        # Get the image's width and height
        image_width, image_height = image.size

        # Calculate the aspect ratio
        aspect_ratio = image_width / image_height

        # Calculate the new width and height
        new_width = min(self.size, aspect_ratio * self.size)
        new_height = min(self.size, self.size / aspect_ratio)

        # Resize the image
        image = image.resize((int(new_width), int(new_height)), Image.BICUBIC)

        # Create a new image with a black background
        new_image = Image.new('RGB', (self.size, self.size), (0, 0, 0))

        # Paste the resized image into the center of the new image
        new_image.paste(image, ((self.size - int(new_width)) // 2, (self.size - int(new_height)) // 2))

        return new_image