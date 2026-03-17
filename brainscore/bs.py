"""
runs an evaluation based on the Brainscore Benchmarks
"""
class BrainScore():
    def __init__(self, model, outdir, device='cpu'):
        self.model = model
        self.outdir = outdir
        self.device = device
    
    def run():
        print("running Brainscore evaluation")