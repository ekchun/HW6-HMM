import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p

        # Making it clearer for me.... define log probabilities already
        self.log_prior = np.log(prior_p)
        self.log_transition = np.log(transition_p)
        self.log_emission = np.log(emission_p)

        # Check that probabilities are valid
        assert np.isclose(np.sum(prior_p), 1)
        assert np.allclose(np.sum(transition_p, axis=1), 1)
        assert np.allclose(np.sum(emission_p, axis=1), 1)


    def _input_check(self, input_observation_states: np.ndarray) -> None:
        """
        
        This function checks that the input observation states are valid!

        """

        if input_observation_states.ndim != 1:
            raise ValueError("Input observation states must be a 1D array.")
        for obs in input_observation_states:
            if obs not in self.observation_states_dict:
                raise ValueError(f"Unknown observation '{obs}'.")

    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        
        self._input_check(input_observation_states) # check that input is valid

        # Step 1. Initialize variables
        T = len(input_observation_states)
        N = len(self.hidden_states)
        if T == 0:
            return 0 # no observations, return 0 probability
        if N == 0: 
            raise ValueError("Need at least 1 hidden state") 
        
        log_forward = np.full((T, N), -np.inf) # create log prob matrix, initialized to -inf for log(0)
        
        first_obs_idx = self.observation_states_dict[input_observation_states[0]]
        log_forward[0] = (self.log_prior + self.log_emission[:, first_obs_idx]) # first row of forward matrix
       
        # Step 2. Calculate probabilities
        # approximation with log-sum-exponent trick to avoid underflow
        for t in range(1, T):
            obs_idx = self.observation_states_dict[input_observation_states[t]]

            for n in range(N):
                prob = (log_forward[t - 1] + self.log_transition[:, n])
                best_prob = np.max(prob)

                log_sum = best_prob + np.log(np.sum(np.exp(prob - best_prob))) # log-sum-exp trick
                log_forward[t, n] = log_sum + self.log_emission[n, obs_idx]

        # Step 3. Return final probability 
        last_log_prob = log_forward[-1]
        best_prob = np.max(last_log_prob)
        log_likelihood = best_prob + np.log(np.sum(np.exp(last_log_prob - best_prob)))

        forward_probability = np.exp(log_likelihood) # convert log
        return forward_probability
        

    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        
        self._input_check(decode_observation_states) # check that input is valid

        # Step 1. Initialize variables
        T = len(decode_observation_states)
        N = len(self.hidden_states)
        if T == 0:
            return [] # no observations, return empty list
        if N == 0: 
            raise ValueError("Need at least 1 hidden state")

        #store probabilities of hidden state at each step - changed to log
        log_viterbi_table = np.full((T, N), -np.inf)
        #store best path for traceback - I changed this to match the Stanford resource I was following
        best_path = np.zeros((T, N), dtype=int)         
       
        first_obs_idx = self.observation_states_dict[decode_observation_states[0]] # index of first observation state
        log_viterbi_table[0] = (self.log_prior + self.log_emission[:, first_obs_idx])

       # Step 2. Calculate Probabilities
        for t in range(1, T):
            obs_idx = self.observation_states_dict[decode_observation_states[t]]

            for n in range(N):
                prob = (log_viterbi_table[t - 1] +self.log_transition[:, n])
                log_viterbi_table[t, n] = np.max(prob) + self.log_emission[n, obs_idx]
                best_path[t, n] = np.argmax(prob) # store index of best previous state for traceback
                    
            if np.all(log_viterbi_table[t] == -np.inf):
                raise ValueError(f"No valid paths at step {t}")

        # Step 3. Traceback 
        best_last_state = np.argmax(log_viterbi_table[-1])
        best_hidden_state_indices = [best_last_state]

        # Step 4. Return best hidden state sequence 
        # Walk backwards through best_path chain
        for t in range(T - 1, 0, -1):
            best_last_state = best_path[t, best_last_state]
            best_hidden_state_indices.insert(0, best_last_state)

        best_hidden_state_sequence = [self.hidden_states[i] for i in best_hidden_state_indices]
    
        return best_hidden_state_sequence
        