import pandas as pd
import numpy as np
import os
from rouge.rouge import Rouge


ref  = 'the only thing crazier than a guy in snowbound massachusetts boxing up the powdery white stuff and offering it for sale online ? people are actually buying it . for $ 89 , self-styled entrepreneur kyle waring will ship you 6 pounds of boston-area snow in an insulated styrofoam box – enough for 10 to 15 snowballs , he says .'
cand = 'a man in suburban boston is selling snow online to customers in warmer states . for $ 89 , he will ship 6 pounds of snow in an insulated styrofoam box .'
rouge = Rouge()
scores = rouge.get_scores(cand, ref)[0]

print(scores)