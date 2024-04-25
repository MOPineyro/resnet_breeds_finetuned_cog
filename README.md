# Dog and Cat Breed Classifier

This repository contains a machine learning model for classifying images of dogs and cats into their respective breeds using a fine-tuned ResNet model. The model is containerized using the Cog library to facilitate easy deployment and usage. For interactive usage and more details, visit the [project page on Replicate](https://replicate.com/mopineyro/resnet_breeds_finetuned).

## Setup

To set up the project, you need to have Docker and [Cog](https://github.com/replicate/cog?tab=readme-ov-file#install) installed on your machine. You will also need to clone this repository to your local machine.

### Prerequisites

- Python 3.8 or later
- Docker
- PyTorch
- torchvision
- PIL
- numpy

### Installation

1. Clone the repository to your local machine.
2. Navigate to the cloned directory.
3. Build the Cog model container:
   ```bash
   cog build
   ```

## Running the Model

You can run the model using the Cog command. To classify an image, use the following command:

```bash
cog predict -i image_url=<URL of image>
```

Replace `<URL of image>` with the actual URL of the image you want to classify.

## Model Details

The model is based on the ResNet18 architecture, adapted to classify 37 different breeds of dogs and cats. The final layer of the model has been modified to output 37 classes, corresponding to these breeds.

The model expects input images in the form of URLs, retrieves them, preprocesses them to fit the model requirements, and then forwards them through the network to obtain a prediction.

### Image Preprocessing

The image preprocessing steps include resizing the image to a fixed dimension (224x224 pixels), converting it to a tensor, and normalizing it with specific mean and standard deviation values used for ResNet models.

## Error Handling

The model includes error handling for scenarios where the image URL might be incorrect or the image cannot be retrieved. It logs appropriate error messages to assist in debugging.

## Learning More

For more information on how to use Cog to containerize and deploy machine learning models, check the [Cog documentation on GitHub](https://github.com/replicate/cog).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.