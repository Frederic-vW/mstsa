"""mstsa.microsynt — syntactic analysis of EEG microstate sequences."""

import matplotlib.pyplot as plt
import numpy as np

from .mstsa import ScalarIntArray


class Microsynt(object):
    """Syntactic analysis of EEG microstate sequences.

    Extracts an optimal vocabulary of entropy-classified words from a
    symbolic sequence, compares their representation against theoretical
    and surrogate distributions, and produces entropy representation
    ratios that characterise the temporal structure of the sequence.

    Parameters
    ----------
    K : int
        Number of symbols (microstate classes).
    w : int
        Word length.
    verbose : bool, optional
        Print progress messages (default ``False``).

    References
    ----------
    .. [1] Artoni, F., Maillard, J., Britz, J., Brunet, D., Lysakowski, C.,
           Tramèr, M.R., & Michel, C.M. (2023). Microsynt: Exploring the
           syntax of EEG microstates. *NeuroImage*, 277, 120196.
           https://doi.org/10.1016/j.neuroimage.2023.120196
    """

    def __init__(self, K: int, w: int, verbose: bool = False) -> None:
        self.K = K  # number of symbols / classes
        self.w = w  # word length
        self.verbose = verbose
        self.states = np.arange(K) # 0, 1, ..., K-1
        self.labels = [str(s) for s in self.states] # list, ['0', '1', '2'...
        self.theoretical_entropy_representation()

    def availables(self, list_: list, element_: str) -> list:
        t = list_.copy()
        t.remove(element_)
        return t

    def empirical_entropy_representation(self) -> None:
        # word entropies of empirical microstate sequence
        hs_emp = np.zeros(self.nt-self.w)
        for t in range(self.nt-self.w):
            # translate 'word' into key and retrieve entropy for that word
            k = self.translate_seq2key(self.data[t:t+self.w])
            hs_emp[t] = self.theo_dict[k]
        # empirical entropy representation (EER)
        self.eer = np.zeros_like(self.ter)
        for i, h in enumerate(self.entropy_classes):
            self.eer[i] = np.sum(hs_emp == h)
        self.eer /= len(hs_emp) # normalize distribution
        if self.verbose:
            print("[+] Empirical entropy representation:")
            print("\t", self.entropy_classes, "\n\t", self.eer)
            print("[+] Empirical Entropy Representation (EER) computed")
            print("\tcheck EER normalization: ", np.sum(self.eer))

    def grow_list(self, l: list) -> list:
        '''
        take list l, assumed to have string elements
        0) initialize empty list v (will be output)
        1) take last element e of l
        2) avoid duplicates of the last char of e, use available elements only
           e.g. if e='ADB' and symbols are 'A,B,C,D,E', availables are
           {A,B,C,D,E} \ B = {A,C,D,E}
        3) append each available symbol to each element of l, push results into v

        Example 1:
        input l = ['A', 'B', 'C', 'D', 'E']
        output = ['EA', 'EB', 'EC', 'ED',
                  'DA', 'DB', 'DC', 'DE',
                  'CA', 'CB', 'CD', 'CE',
                  'BA', 'BC', 'BD', 'BE',
                  'AB', 'AC', 'AD', 'AE']

        Example 2:
        input l = ['EA', 'EB', 'EC', 'ED',
                   'DA', 'DB', 'DC', 'DE',
                   'CA', 'CB', 'CD', 'CE',
                   'BA', 'BC', 'BD', 'BE',
                   'AB', 'AC', 'AD', 'AE']
        output =  ['AEA', 'AEB', 'AEC', 'AED',
                   'ADA', 'ADB', ...

        '''
        v = [] # new list
        n = len(l)
        for i in range(n):
            e = l.pop() # pop last element from list l
            available_elements = self.availables(self.labels, e[-1])
            new_elements = [e+f for f in available_elements]
            v = v + new_elements
        return v

    def load_sequence(self, filename: str, preproc: bool = False) -> None:
        # load microstate sequence, permanence or no-permanence
        if self.verbose:
            print(f"[+] Loading file: {filename:s}")
        self.data = np.load(filename)
        self.nt = len(self.data)

    def markov_entropy_representation(self) -> None:
        """
        Determine the Markov chain probability of each word

        """
        if self.verbose:
            print("\n[+] MC word probabilities")

        # class distribution
        p = np.array([np.sum(self.data==k)/self.nt for k in range(self.K)])
        self.p = p
        if self.verbose:
            print("[+] p = ", p)

        # Markov-1 surrogate: compute the transition matrix of self.data
        T1 = np.zeros((self.K,self.K))
        for i in range(self.nt-1):
            T1[self.data[i],self.data[i+1]] += 1
        rowsum = T1.sum(axis=1, keepdims=True)
        rowsum[rowsum==0] = 1
        T1 /= rowsum # corrects for potential zero rows
        self.T1 = T1

        # surrogate entropy representation
        mer = np.zeros(self.n_entropy_classes)
        p_mc1_sum = 0.0 # denominator for normalization
        # take each word w from the theoretical dictionary and compute P_MC(w)
        # P(w) = P(w[0]) * P(w[1] | |w[0]) * P(w[2] | |w[1]) * ...
        for word in self.theo_dict:
            s = self.translate_key2seq(word)
            p_mc1 = p[s[0]] * np.prod([T1[i,j] for i,j in zip(s[:-1], s[1:])])
            p_mc1_sum += p_mc1
            # probability-weighted sum of words
            h = self.theo_dict[word]
            l = self.entropy_classes.tolist().index(h) # index of entropy class
            mer[l] += p_mc1
        # representation ratio
        # p_mc1_sum must sum to 1.0 as all possible words are checked
        mer /= p_mc1_sum
        # Markov entropy representation ratios
        mc1_err = self.eer/mer
        self.mer = mer

    def plot_representation_ratio(self) -> None:
        hs = self.entropy_classes
        fig, ax = plt.subplots(1, 2, figsize=(16,4))

        ax[0].plot(hs, self.ter, '-ok',
                   label="Theoretical Entropy Representation")
        ax[0].plot(hs, self.eer, '-sb',
                   label="Empirical Entropy Representation")
        ax[0].set_xlabel("entropy (bits)")
        ax[0].set_ylabel("theo. prob.")
        ax[0].legend()

        ax[1].semilogy(hs, self.eer/self.ter, '-ok')
        ax[1].semilogy(hs, np.ones_like(hs), '--k', lw=2)
        ax[1].set_xlabel("entropy (bits)")
        ax[1].set_ylabel("theo. prob.")
        ax[1].set_title("Entropy representation ratio")

        plt.tight_layout()
        plt.show()

    def set_sequence(self, x: ScalarIntArray) -> None:
        self.data = x  # set discrete-valued sequence
        self.nt = len(self.data)

    def theoretical_entropy_representation(self) -> None:
        """Build the theoretical entropy representation dictionary.

        Enumerates all non-repeating words of length ``self.w`` over
        ``self.K`` symbols, computes each word's Shannon entropy, and
        stores the result in ``self.theo_dict`` and related attributes.
        """
        # create word list, no duplicates
        words = self.labels.copy()
        for _ in range(self.w-1):
            words = self.grow_list(words)
        # check that the correct number of words has been found
        K = self.K
        w = self.w
        N = K*(K-1)**(w-1)
        assert N == len(words)
        if self.verbose:
            print(f"[+] Check: theoretical N = {N:d}, real N = {len(words):d}")

        # theoretical dictionary
        self.theo_dict = {}
        hs = np.zeros(N) # theoretical entropies
        for i, word in enumerate(words):
            h = self.word_entropy(word) # unrounded
            h = np.round(h, decimals=3) # rounded, for easy class assignment
            hs[i] = h
            self.theo_dict[word] = h
        # entropy classes
        self.entropy_classes = np.sort(np.unique(hs))
        self.n_entropy_classes = len(self.entropy_classes)

        # theoretical entropy representation (TER)
        self.ter = np.zeros(self.n_entropy_classes)
        for i, h in enumerate(self.entropy_classes):
            self.ter[i] = np.sum(hs == h)
        self.ter /= len(hs) # normalize distribution

    def translate_key2seq(self, word: str) -> list:
        """
        Translate microstate 'word' to integer sequence
        Example: '021321' -> [0,2,1,3,2,1]
        """
        s = [int(c) for c in word]
        return s

    def translate_seq2key(self, x: ScalarIntArray) -> str:
        """
        Translate microstate sub-sequence into string
        Example: [0,2,1,3,2,1] -> '021321'
        """
        s = "".join([f"{y:d}" for y in x])
        return s

    def word_entropy(self, word: str, base: str = '2') -> float:
        word_ints = [int(c) for c in word] # integers
        _log = np.log2 if base == '2' else np.log
        p = np.zeros(self.K)
        for i in word_ints:
            p[i] += 1.0
        p /= len(word_ints) # normalize histogram
        h = -np.sum(p[p>0]*_log(p[p>0]))
        return h
