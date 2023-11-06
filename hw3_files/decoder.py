from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

from extract_training_data import FeatureExtractor, State

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1, len(words)))
        state.stack.append(0)

        while state.buffer:
            features = self.extractor.get_input_representation(words, pos, state)
            predictions = self.model.predict(np.array([features]))[0]
            sorted_action_indices = np.argsort(predictions)[::-1]
            action, label = self.output_labels[sorted_action_indices[0]]
            # print(action)
            # print("state",state)
            # sys.exit()
            if action == 'shift':
                state.shift()
            elif action == 'left_arc' and not (state.stack and state.stack[-1] == 0):
                state.left_arc(label)
            elif action == 'right_arc' :
                state.right_arc(label)

        result = DependencyStructure()
        for p, c, r in state.deps:
            result.add_deprel(DependencyEdge(c, words[c], pos[c], p, r))
        return result


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)

    # argv = ['', 'data/model.h5', 'data/dev.conll']
    # parser = Parser(extractor, argv[1])
    # with open(argv[2],'r') as in_file: 
    #     for dtree in conll_reader(in_file):
    #         words = dtree.words()
    #         pos = dtree.pos()
    #         deps = parser.parse_sentence(words, pos)
    #         print(deps.print_conll())
    #         print()

    parser = Parser(extractor, sys.argv[1])
    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
        
