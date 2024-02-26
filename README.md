# Sporty

Sporty is an AI model designed to assist during workout sessions by monitoring repetitions and movements using the Mediapipe library.

![Sporty Demo](https://mir-s3-cdn-cf.behance.net/project_modules/max_1200/91c71e44324263.58171f4cc22e3.gif)

## Environment and Requirements

Ensure you have Python 3.10.6 installed before proceeding.

1. Create a virtual environment using Python 3 (follow the installation instructions at [virtualenvwrapper documentation](https://virtualenvwrapper.readthedocs.io/en/latest/install.html)):

    ```
    virtualenv <name_env>
    ```

2. Activate the virtual environment:

    ```
    source <name_env>/bin/activate
    ```

3. Install Python requirements:

    ```
    pip install -r requirements.txt
    ```

**Important**: The latest [PyAV](https://pypi.org/project/av/) versions are incompatible with the pytorch version used. Info: https://github.com/pytorch/vision/issues/7305


## Data

Sporty utilizes the [Workout Video dataset](https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video) from Kaggle, which consists of 22 different categories of workouts.

## Usage

Sporty offers two modes: with or without exercise prediction.

Run the main script:

```
python main.py
```

### Without Exercise Prediction

In this mode, it's not necessary to have the model pre-trained beforehand. Currently active exercises include:

- Pull Up
- Squat

### Exercise Prediction

To use this mode, where Sporty predicts exercises, it's necessary to train the model first. Follow these steps:

1. **Preprocess Videos**: Run `preprocess_videos.py`.
2. **Split Data**: Run `split_data.py`.
3. **Train Model**: Run `train.py`.

## Next Steps

- Incorporate correction mechanisms for new exercises.
- Ensure predicted exercise names match the defined exercises.

Feel free to contribute and enhance Sporty's capabilities!

