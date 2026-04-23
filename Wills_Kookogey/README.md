hi guys.
welcome to fennec
if you've made it this far, congrats on learning something about git
(or just doing what chat told you to, or mega congrats if you actually knew how to git prior to this)

this is the Fault Identification (FID) repository. I left a lot of my codes in here cause I wanted whoever's reading this to know how much
trial and error this all takes.
For reference, the codes left behind are only a fraction of the many renditions and revisions.
its fun
unless it breaks. then its not

anyways enough chatter, I wouldn't advise trying to run any of these jupyter notebooks except for the ones called "FID4_..."
The earlier codes used different data processing functions which don't exist anymore cause they were faulty and caused data leakage.
Those functions were deleted so that (hopefully) only good examples can be found in this repo of how to do all this stuff. But the jupyter
notebooks were left behind as a memorial to the time lost on bad solutions. it's just part of the process.

pro tip: its all in the data processing. if your model isn't working, it's probably not your model's fault, it's your data. (or big
picture model architecture, ie LSTM vs GRU vs convolutional, etc) but with good data *processed the right way*, you should have lots of
success. And that's when it's fun

FID4_3 was the model presented to the navy spring 2026, and is the most consistently good model (avg 84% ish accuracy, pretty consistent
across all classes). accuracy will vary within multiple percents each time you train a model, so if you run this code and train this model
it may be slightly better or worse.

FID4_5 and higher experimented with a sliding window on the training dataset, to give the model more training data, and this improved
overall accuracy, but for some reason it made the LR category really inaccurate. bleh. anyways still not fully over that one.

some of this info is repeated elsewhere, but here are some bullet points that should encapsulate everything you need to do to get this
code up and running on your machine 

- get jupyter notebooks running (and learn the keyboard shortcuts to run cells, that'll save your time and wrists)
- install all the dependencies listed in the README at the root folder FENNEC-25_26
- download the DATA folder from onedrive, and move it into the Wills_Kookogey folder.
- (the FID_utils.py file has some helpful data processing functions which these jupyter notebooks need to run)

if something is broken, ask chat or text me (615-864-5991) or email me (willskookogey@gmail.com)


(oh btw i did this all on my mac (what a beast for what it is), so some specific lines such as the pytorch torch.device lines will have to be reconfigured to match the system youre running on. just pop the codes into chat or claude and they'll give you the few alterations you'll need to make)