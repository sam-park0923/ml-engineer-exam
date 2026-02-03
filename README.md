# README #

* Quick Summary

The Milliman IntelliScript Machine Learning Engineer Exam.

## Repo Overview ##

* Folder Summary

- src
  - ml_engineer_exam
    - model
      - The module needed to train and evaluate the model
    - prediction
      - The module needed to make predictions with the trained model
    - prepare
      - The module needed to prepare the data for training and evaluation
    - scripts
      - run_model_training.py
        - Trains a model ('linear', 'ridge', 'random_forest') using the California housing dataset.
      - run_prediction.py
        - Makes predictions using a trained model and sample input data.
    - config.py
      - Config classes for model training and prediction

### How do I get set up? ###

* Pre-requisites (local running)
  - [Setup SSH Keys needed to pull down repositories](https://www.atlassian.com/git/tutorials/git-ssh)
  - [Install UV](https://docs.astral.sh/uv/getting-started/installation/)

* Repo-setup

- Clone Repo (in IDE)
- Setup UV Environment

### How to run ###

#### Run scripts ####

  - Run Command
  ```shell
  uv run run_model_training --model_type linear 
  uv run run_prediction--model_name linear --input_data "{\"MedInc\": 1.6812, \"HouseAge\": 25.0, \"AveRooms\": 4.192200557103064, \"AveBedrms\": 1.0222841225626742, \"Population\": 1392.0, \"AveOccup\": 3.877437325905293, \"Latitude\": 36.06, \"Longitude\": -119.01}"
  ```

### Run Tests ###

#### Run Unit Tests For Chart Summary ####

```shell
uv run pytest -v
```

### Contribution guidelines ###

* Code review

All code reviews should be attached to a merge request or equivalent in your version control system 
(e.g. merge requests are called pull requests in bitbucket)

* Other guidelines

- Add doc strings (preferable restStructuredText)
- Use an IDE like Pycharm, Visual Studio Code, 
- Follow PEP standards
- Create new branches for any work that you do
- Make sure to bump the project version

  ```bash
  uv version --bump minor #patch or minor or major (0.0.1 or major.minor.patch)
  ```

### Who do I talk to? ###

* Dependencies

The project dependencies are located in the pyproject.toml file.
You can see them by running a pip command "pip show ml_engineer_exam" after installing the package via uv.

* Repo owner or admin 

Contact nicholas.arquette@milliman.com