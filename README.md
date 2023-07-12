# meeting-summary

This script is designed to take in a m4a zoom audio recording and with the power of OpenAI API and langchain it uses AI to both transcribe and summarize the meeting.

## How to run the script
Created using Python 3.9.2

Use the `pip install -r requirements.txt` command to install all of the Python modules and packages listed in the requirements.txt file

```OPENAI_API_KEY=YOUR-KEY-HERE python meeting_summary.py NAME_OF_YOUR_ZOOM_AUDIO_RECORDING.m4a```

It should then print the meeting summary to the console when done

## Todo
At the moment the script makes a mess of the working directory by adding all the text files and wav files to it. I'd like to clean that up in the future
