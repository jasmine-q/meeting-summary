import os
import sys
import math
import openai
from pydub import AudioSegment
from pydub.utils import make_chunks
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.tools.file_management import (
    ReadFileTool,
    CopyFileTool,
    DeleteFileTool,
    MoveFileTool,
    WriteFileTool,
    ListDirectoryTool,
)
from langchain.agents.agent_toolkits import FileManagementToolkit

m4a_source = sys.argv[1]

print("converting and splitting meeting audio file: ", m4a_source)

# convert audio to wav
# and split the audio file into less than 25mb chunks for OpenAI API file size restrictions

m4a_audio = AudioSegment.from_file(m4a_source, format='m4a')
m4a_audio.export("audio.wav", format="wav")

audio = AudioSegment.from_file("audio.wav" , "wav")

audio = audio.set_channels(1)       # mono
audio = audio.set_frame_rate(16000) # 16000Hz
channel_count = audio.channels      # Get channels
sample_width = audio.sample_width   # Get sample width
duration_in_sec = len(audio) / 1000 # Length of audio in sec
sample_rate = audio.frame_rate
bit_rate = 16

wav_file_size = (sample_rate * bit_rate * channel_count * duration_in_sec) / 8

# max file size of 25mb
file_split_size = 25000000
total_chunks =  wav_file_size // file_split_size

# Get chunk size by following method
# for  duration_in_sec (X) -->  wav_file_size (Y)
# So   whats duration in sec  (K) --> for file size of 10Mb
# K = X * 10Mb / Y

chunk_length_in_sec = math.floor((duration_in_sec * file_split_size ) /wav_file_size) # in sec
chunk_length_ms = chunk_length_in_sec * 1000
chunks = make_chunks(audio, chunk_length_ms)

# Export all of the individual chunks as wav files

for i, chunk in enumerate(chunks):
  chunk_name = "chunk{0}.wav".format(i)
  print("exporting wav audio chunks", chunk_name)
  chunk.export(chunk_name, format="wav")


# setup openai and langchain
openai.api_key = os.getenv("OPENAI_API_KEY")
tools = FileManagementToolkit(
    selected_tools=["read_file", "write_file", "list_directory"],
).get_tools()
read_tool, write_tool, list_tool = tools

# For each chunk wav file, transcribe it and write the transcript to a text file
print("starting meeting transcription, this might take awhile...")
for i, chunk in enumerate(chunks):
  audio_file = open("chunk{0}.wav".format(i), "rb")

  # haven't figured out how to use the transcription using langchain yet
  transcript = openai.Audio.transcribe(
    file = audio_file,
    model = "whisper-1",
    response_format="text",
    language="en"
  )
  write_tool.run({"file_path": "chunk%s.txt" % i, "text": transcript})


llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
loader = DirectoryLoader('.', glob="chunk*.txt", loader_cls=TextLoader)

docs = loader.load()

prompt_template = """Write a summary that goes over the key points of this meeting transcript:
Transcript:
{text}
"""
refine_template = (
    "Your job is to produce a final summary\n"
    "We have provided an existing summary up to a certain point: {existing_answer}\n"
    "We have the opportunity to add more to the existing summary"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{text}\n"
    "------------\n"
    "Given the new context, refine the original summary"
    "If the context isn't useful, return the original summary."
)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
refine_prompt = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template,
)
chain = load_summarize_chain(llm, chain_type="refine", question_prompt=PROMPT, refine_prompt=refine_prompt)

print("starting langchain meeting summarisation...")
print(chain.run(docs))
