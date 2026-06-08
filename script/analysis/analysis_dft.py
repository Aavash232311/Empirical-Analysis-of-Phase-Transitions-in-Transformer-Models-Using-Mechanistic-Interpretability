import torch
from utils.explained_variance_ratio import heat_map_raw_lattice



''' 
WE ARE INITILIZING ANALYSIS ABOVE THE EPOCH.
AND THEN USING THIS SAME CLASS, BE CAREFUL WITH THE DEFINING THE 
VALUES IN THE CONSTRUCTOR, MEMORY WILL LEAK.

'''
class Analysis:

    def __init__(self, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = kwargs.get("model")
        skeleton = kwargs.get("skeleton")
        vocab_size = kwargs.get("vocab_size")
        data = kwargs.get("data")

        
        self.vocab_size = vocab_size


        if 'model' in kwargs:
            self.model = model.to(self.device)

        '''' 
        Analysis of diagonal spectral mass post training.
        skeleton and dataset is required.

        Analysis during the training leave
        skeleton and data as None
        
        '''
        if skeleton is not None and data is None: # else we might want to evulate the diagonal spectral mass during the training.
            self.model.load_state_dict(skeleton)
        self.data = data

    ''' Expects two arguments preferred_logits and x if we want to analysis during the training. '''
    def discrete_fourier_transform(self, **kwargs):
        n = self.vocab_size
        logit_lattice = torch.zeros(n, n, n, device=self.device)

    
        ''' 
            Here we are doing a forward pass for logits making sure we don't have randomness
            from for example dropout.

            We need logits_lattice in n n n dimnesion to add see I will make this method here.
        '''

        if 'preferred_logits' in kwargs:
            logits = kwargs.get("preferred_logits")
            x = kwargs.get("x")
            ''' 
                IF CHECKING DURING TRAINING THEN LOGITS FORM TRANSFORMER DIRECTLY
            '''
            for i in range(x.shape[0]):
                a = x[i, -2].item()  
                b = x[i, -1].item() 
                logit_lattice[a, b] = logits[i, -1, :] 

        else:
            '''
                IF PRE TRAINED MODEL THEN WE DIRECTLY FORWARD PASS
            '''
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in self.data:
                    x = x.to(self.device)
                    logits = self.model(x)  

                    for i in range(x.shape[0]):
                        a = x[i, -2].item()   # second to last token
                        b = x[i, -1].item()   # last token
                        logit_lattice[a, b] = logits[i, -1, :] 

                    pred = logits[:, -1, :].argmax(dim=-1)     
                    expected = (x[:, -2] + x[:, -1]) % self.vocab_size  
                    correct += (pred == expected).sum().item()
                    total += x.shape[0]
    
            print(f"correct: {correct}/{total} = {correct/total*100:.1f}%")
            print(f"unique (a,b) pairs seen: {(logit_lattice.sum(dim=-1) != 0).sum().item()}")
 
 
        # ℓ̃ (c|a,b) = ℓ(c|a,b) - (1/n) Σ ℓ(c'|a,b) centered logits
        logit_lattice = logit_lattice - logit_lattice.mean(dim=-1, keepdim=True)

        n = self.vocab_size

        fourier_lattice = torch.zeros(n, n, n, dtype=torch.complex64)

 
        for c in range(n):
            phi = logit_lattice[:, :, c] # shape [n, n] one class slice
            fourier_lattice[:, :, c] = torch.fft.fft2(phi) / n   # eq. 17

        power_spectrum = fourier_lattice.abs() ** 2  # shape [n, n, n] eqn 16, basically squaring

        return power_spectrum
        
    ''' 
        The 'preferred_logits' is passed when we want to analyse 
        the the evolution of diagonal spectral mass duing each epoch.

    '''
    def diagonal_sperectal_mass(self, **kwargs):


        if 'preferred_logits' in kwargs:
            power_spectrum = self.discrete_fourier_transform(preferred_logits=kwargs.get("preferred_logits"), x=kwargs.get("x"))
        else:
            power_spectrum = self.discrete_fourier_transform()

        # we need to reduce the dimension adding third col, to first and second
        # we want to know total power at each frequency

        ''' 
        mass of a particle <-> how concentrated the energy is
        light particle = spread out wave
        heavy particle = localized, concentrated 
        '''


        n = self.vocab_size
        S = power_spectrum.sum(dim=-1)
        numerator = sum(S[m, m] for m in range(1, n))
        denominator = S.sum() - S[0, 0] # expect the 0,0 as in formula
        m_theta = (numerator / denominator).item()
        
        return m_theta
    
    ''' This is not functional right now. 
        We can remove this.
    '''
    def heat_map(self):
        heat_map_raw_lattice(self.uncentered_lattice, self.vocab_size) # it's centered a unchanged after that

        