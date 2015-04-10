#ex19.py
#-------
from composes.semantic_space.space import Space
from composes.composition.lexical_function import LexicalFunction
from composes.utils.regression_learner import LstsqRegressionLearner

#training data1: VO N -> SVO
train_vo_data = [("hate_boy", "man", "man_hate_boy"),
                 ("hate_man", "man", "man_hate_man"),
                 ("hate_boy", "boy", "boy_hate_boy"),
                 ("hate_man", "boy", "boy_hate_man")
                 ]

#training data2: V N -> VO
train_v_data = [("hate", "man", "hate_man"),
                ("hate", "boy", "hate_boy")
                ]

#load N and SVO spaces
n_space = Space.build(data = "./data/in/ex19-n.sm",
                      cols = "./data/in/ex19-n.cols",
                      format = "sm")

svo_space = Space.build(data = "./data/in/ex19-svo.sm",
                        cols = "./data/in/ex19-svo.cols",
                        format = "sm")

print "\nInput SVO training space:"
print svo_space.id2row
print svo_space.cooccurrence_matrix

#1. train a model to learn VO functions on train data: VO N -> SVO
print "\nStep 1 training"
vo_model = LexicalFunction(learner=LstsqRegressionLearner())
vo_model.train(train_vo_data, n_space, svo_space)

#2. train a model to learn V functions on train data: V N -> VO
# where VO space: function space learned in step 1
print "\nStep 2 training"
vo_space = vo_model.function_space
v_model = LexicalFunction(learner=LstsqRegressionLearner())
v_model.train(train_v_data, n_space, vo_space)

#print the learned model
print "\n3D Verb space"
print v_model.function_space.id2row
print v_model.function_space.cooccurrence_matrix


#3. use the trained models to compose new SVO sentences

#3.1 use the V model to create new VO combinations
vo_composed_space = v_model.compose([("hate", "woman", "hate_woman"),
                                     ("hate", "man", "hate_man")],
                                    n_space)

#3.2 the new VO combinations will be used as functions:
# load the new VO combinations obtained through composition into
# a new composition model
expanded_vo_model = LexicalFunction(function_space=vo_composed_space,
                                    intercept=v_model._has_intercept)

#3.3 use the new VO combinations by composing them with subject nouns
# in order to obtain new SVO sentences
svo_composed_space = expanded_vo_model.compose([("hate_woman", "woman", "woman_hates_woman"),
                                                ("hate_man", "man", "man_hates_man")],
                                                n_space)

#print the composed spaces:
print "\nVO composed space:"
print vo_composed_space.id2row
print vo_composed_space.cooccurrence_matrix

#print the composed spaces:
print "\nSVO composed space:"
print svo_composed_space.id2row
print svo_composed_space.cooccurrence_matrix

