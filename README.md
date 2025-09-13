# AI_STT
this repository is for speech to text transcription
## Repository files

**main.py**: script of main service.

**test.py**: you can test script in develop mod with this script.

**test.wav**: test wav file used in test.py

**.env(optional)**: define environment variables in this file


## parameters

**HOST**: service host that you can assign in environment variable..

**PORT**: serice port that you can assign in environment variable.


# Install .venv
you can install virtual environment with **poetry**

        poetry install --no-root


then run entrypoint to config **LD_LIBRARY_PATH**


        source entrypoint.sh

# Run main.py and test.py
if you use poetry you can run main.py with:

        poetry run python main.py

or you can run the main.py on host 0.0.0.0 and port 5001 with

        poetry run uvicorn main:app --host 0.0.0.0 --port 5001 --reload 
also you can test your developement main.py with running test.py 

with:

        poetry run python main.py




# Docker
you can build docker image and run a container with the image 
Docker file Expose port is **8000**
with:


        docker build --no-cache -t stt-server .
        docker run -it --gpus all -p 5001:8000  stt-server


# Input/Output structure
input and output of the service in dictionary structure.

there is two parameter **("type", "audio")**
in input stage type should be "input_audio_buffer.append" 

and in output types are two **("error", "speech_transcription")**
audio file should be ***"base64"***

## Input structure:

        {
            "type": "input_audio_buffer.append", 
            "sample_rate": Default(16000)
            "audio": audio_b64
        }

## output structure:
there is 2 state or type in output structure


in "Error":

        {
            "type": "error", 
            "message": "Server models not initialized."
        }


in "speech_transcription":


        {
            "type": "response.audio_transcript.done",
            "audio": segment_b64,
            "sample_rate": VAD_SAMPLE_RATE
        }
