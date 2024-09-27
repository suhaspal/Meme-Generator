This meme generator was created by fine-tuning a CompVis/stable diffusion model from HuggingFace over a large dataset of images and fine tuning a DistilBERT model from HuggingFace to train over a dataset of jokes.

The two models are connected together by the use of Langgraph and AgentState, through which I am able to feed inputs and gather outputs through every stage of the program.

Because I was not able to have access to a CUDA enabled GPU, I tried my best to train for as long as possible on an M2 Pro Mac's CPU, but the results were definitely underwhelming. I believe with more time,
the models could have been trained much better and have produced better results.

I have included all the code that was used in this process. I tried to upload the generated models as well, but I seem to have hit the file size limit on GitHub. All of the models can be generated using the code provided in this repository.
