import pytest
from hmm import HiddenMarkovModel
import numpy as np


def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')

    hmm = HiddenMarkovModel(
        observation_states=mini_hmm["observation_states"],
        hidden_states=mini_hmm["hidden_states"],
        prior_p=mini_hmm["prior_p"],
        transition_p=mini_hmm["transition_p"],
        emission_p=mini_hmm["emission_p"])

    observation_seq = mini_input["observation_state_sequence"]
    correct_hs_seq = mini_input["best_hidden_state_sequence"]

    forward_prob = hmm.forward(observation_seq)
    viterbi_path = hmm.viterbi(observation_seq)

    # check that forward is correct
    # probability must be between 0 and 1
    assert forward_prob >= 0
    assert forward_prob <= 1
    # check that forward probability is close to expected value (floats)
    assert np.isclose(forward_prob, hmm.forward(observation_seq), atol=1e-10)

    # check that viterbi path is correct
    assert len(viterbi_path) == len(observation_seq)
    assert list(viterbi_path) == list(correct_hs_seq)

    # edge cases: empty sequence
    empty_seq = np.array([])
    assert hmm.viterbi(empty_seq) == [] or ValueError("Need at least 1 hidden state")
    assert hmm.forward(empty_seq) == 0 or ValueError("Need at least 1 hidden state")

    # edge cases: single observation
    single_obs = np.array([observation_seq[0]])
    forward_prob_single = hmm.forward(single_obs)
    viterbi_single = hmm.viterbi(single_obs)
    # should contain exactly 1 state
    assert len(viterbi_single) == 1
    # forward prob should still be between 0 and 1
    assert 0 <= forward_prob_single <= 1

    # edge cases: unknown observation
    unknown_obs = np.array(["unknownn"])
    with pytest.raises(ValueError):
        hmm.viterbi(unknown_obs)
    with pytest.raises(ValueError):
        hmm.forward(unknown_obs)

    #edge cases: all observations the same
    same_obs_seq = np.array([observation_seq[0]] * len(observation_seq))
    forward_prob_same = hmm.forward(same_obs_seq)
    viterbi_same = hmm.viterbi(same_obs_seq)

    assert len(viterbi_same) == len(same_obs_seq)
    assert 0 <= forward_prob_same <= 1

    for state in viterbi_same:
        assert state in hmm.hidden_states


def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """
    full_hmm = np.load('./data/full_weather_hmm.npz')
    full_input = np.load('./data/full_weather_sequences.npz')

    hmm = HiddenMarkovModel(
        observation_states=full_hmm["observation_states"],
        hidden_states=full_hmm["hidden_states"],
        prior_p=full_hmm["prior_p"],
        transition_p=full_hmm["transition_p"],
        emission_p=full_hmm["emission_p"])

    observation_seq = full_input["observation_state_sequence"]
    correct_hs_seq = full_input["best_hidden_state_sequence"]
    viterbi_path = hmm.viterbi(observation_seq)

    # check that viterbi path is correct
    assert len(viterbi_path) == len(observation_seq)
    assert list(viterbi_path) == list(correct_hs_seq)

    # check that all states in viterbi path are valid hidden states
    for state in viterbi_path:
        assert state in hmm.hidden_states











