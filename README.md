# Resume Parser

Resume Parser is a Python project for getting information from a resume.

## Feature
This resume parser is able to extract **NAME, **EMAIL, and **PHONE_NUMBER.
 
## Installation

1. clone the project from Github
2. create a virtual environment and activate it.

```bash
virtualenv .env
source .env/bin/activate
```
3. install required packages.
```bash
pip install -r requirements.txt
```
4. Run the project.
```bash
python main.py
```



## Run the project using docker

1. build the docker image through Dockerfile.
```bash
docker build -t resume .
```
2. Run the docker.
```bash
docker run resume
```
## Future Work
1. we can add more feature like education, experience, skills, etc.
