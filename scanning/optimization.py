
class Simulation():

    max_acc = None     # above this acceleration, hits are removed
    convolve = None    # apply convolution
    pols_on = None     # which initial polarizations are on 
    # wafers_on, rhombi_on 
    norm_time = None   # normalize to get hits/px/sec

    px_list = None # performing stats on these pixels
    norm_pxan = None    

    def __init__(self) -> None:
        # pass TelescopePattern, Instrument
        self._simulate_scan()
        self._stats()

    def _simulate_scan(self):
        sky_hist = None

        # for single-pixel analysis:
        det_hist = None
        time_hist = None

        # maybe more 

    def _stats(self):
        # generate metrics for optimization (if not already)
        pass

    @property
    def std_dev(self):
        pass

    

# plotting functions