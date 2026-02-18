# HW6-HMM

In this assignment, you'll implement the Forward and Viterbi Algorithms (dynamic programming). 


# Eden's Notes
Used https://web.stanford.edu/~jurafsky/slp3/A.pdf for forward algorithm, super helpful. https://www.cs.cmu.edu/~mgormley/courses/10601-s23/handouts/hw7_recitation_solution.pdf for help with log calculations and "log-sum-exponent trick' that I was seeing online (p. 17). And https://pieriantraining.com/viterbi-algorithm-implementation-in-python-a-practical-guide/ helpful as well.

  * Do your model probabilites add up to the correct values? Is scaling required?
    **Put into log scale jic but I mean I'm not sure it's requried for our test cases?**
  * How will your model handle zero-probability transitions? 
    **Raises ValueError if there's no path**
  * Are the inputs in compatible shapes/sizes with each other? 
    **Added private method (is that the correct term?) to do an initial input check**
  * Any other edge cases you can think of?
  * Ensure that your code accomodates at least 2 possible edge cases. 
    **Considered empty input sequence into forward and viterbi; considered empty hidden states array; added assertions to init to check that probability distributions look good... do these count.... also lots of extra edge case tests within the pytest. **


# Assignment

## Overview 

The goal of this assignment is to implement the Forward and Viterbi Algorithms for Hidden Markov Models (HMMs).

## Tasks and Data 
Please complete the `forward` and `viterbi` functions in the HiddenMarkovModel class. 

We have provided two HMM models (mini_weather_hmm.npz and full_weather_hmm.npz) which explore the relationships between observable weather phenomenon and the temperature outside. Start with the mini_weather_hmm model for testing and debugging. Both include the following arrays:
* `hidden_states`: list of possible hidden states 
* `observation_states`: list of possible observation states 
* `prior_p`: prior probabilities of hidden states (in order given in `hidden_states`) 
* `transition_p`: transition probabilities of hidden states (in order given in `hidden_states`)
* `emission_p`: emission probabilities (`hidden_states` --> `observation_states`)

For both datasets, we also provide input observation sequences and the solution for their best hidden state sequences. 
* `observation_state_sequence`: observation sequence to test 
* `best_hidden_state_sequence`: correct viterbi hidden state sequence 

Create an HMM class instance for both models and test that your Forward and Viterbi implementation returns the correct probabilities and hidden state sequence for each of the observation sequences.

## Task List

[TODO] Complete the HiddenMarkovModel Class methods  <br>
  [x] complete the `forward` function in the HiddenMarkovModelClass <br>
  [x] complete the `viterbi` function in the HiddenMarkovModelClass <br>

[TODO] Unit Testing  <br>
  [x] Ensure functionality on mini and full weather dataset <br>
  [x] Account for edge cases 

[TODO] Packaging <br>
  [x] Update README with description of your methods <br>
  [ ] pip installable module (optional)<br>
  [ ] github actions (install + pytest) (optional)


## Completing the Assignment 
Push your code to GitHub with passing unit tests, and submit a link to your repository [here](https://forms.gle/xw98ZVQjaJvZaAzSA)

### Grading 

* Algorithm implementation (6 points)
    * Forward algorithm is correct (2)
    * Viterbi is correct (2)
    * Output is correct on small weather dataset (1)
    * Output is correct on full weather dataset (1)

* Unit Tests (3 points)
    * Mini model unit test (1)
    * Full model unit test (1)
    * Edge cases (1)

* Style (1 point)
    * Readable code and updated README with a description of your methods 

* Extra credit (0.5 points)
    * Pip installable and Github actions (0.5)
